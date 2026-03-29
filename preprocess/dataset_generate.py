#!/usr/bin/env python
# coding=utf-8
"""
pipeline_detect_mask_caption.py  —  4-Phase Pipeline
═══════════════════════════════════════════════════════════════════════════════
Phase 0 — Prepare dataset images:
    • max(w, h) < min_side  → skip (too small)
    • max(w, h) <= max_side → copy as-is  → dataset/images/
    • max(w, h) >  max_side → resize to max_side (keep ratio) → dataset/images/
    Records scale factor per image for bbox re-mapping.

Phase 1 — OWLv2 Detect (on original image) + SAM2 Mask (on dataset image):
    • OWLv2 runs on ORIGINAL image  → bbox_original
    • Scale bbox  →  bbox_dataset   (using scale factor from Phase 0)
    • SAM2 runs on DATASET image + bbox_dataset → mask (same size as dataset image)
    • Save dataset/masks/

Phase 2+3 — Qwen3VL Caption:
    • Qwen3VL runs on ORIGINAL image + bbox_original → best caption
    • Filter sentences → build SDXL prompt
    • Write detection_results.json

Output JSON fields per entry:
    image_file, mask_file, object_label, prompt
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

# ═══ 1. MOCK FLASH_ATTN — TRƯỚC MỌI THỨ ═══
import sys, types, importlib.util

def _make_mock_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    mod.__version__ = "0.0.0"
    return mod

for _mod_name in [
    "flash_attn",
    "flash_attn.bert_padding",
    "flash_attn.flash_attn_interface",
    "flash_attn.modules",
    "flash_attn.modules.mha",
]:
    sys.modules[_mod_name] = _make_mock_module(_mod_name)

# ═══ 2. PATCH TRANSFORMERS TRƯỚC KHI NÓ LOAD ═══
import transformers.utils.import_utils as _tf_import
if hasattr(_tf_import, "PACKAGE_DISTRIBUTION_MAPPING"):
    _tf_import.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])

# ═══ 3. BÂY GIỜ MỚI IMPORT CÒN LẠI ═══
import argparse
import gc
import json
import os
import shutil
import time
import types
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


# ─────────────────────────────────────────────────────────────────────────────
#  Bypass flash_attn (not always installed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return mod

for _mod_name in ["flash_attn", "flash_attn.bert_padding"]:
    sys.modules[_mod_name] = _make_mock_module(_mod_name)


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ═════════════════════════════════════════════════════════════════════════════
#  MEMORY MONITORING
# ═════════════════════════════════════════════════════════════════════════════

def get_memory_usage() -> tuple[float, float]:
    """Returns (RAM_percent, RAM_GB_used)."""
    proc = psutil.Process()
    rss_bytes = proc.memory_info().rss
    total_mem = psutil.virtual_memory().total
    pct = (rss_bytes / total_mem) * 100
    gb = rss_bytes / (1024**3)
    return pct, gb

def check_memory(threshold_pct: float = 85.0, prefix: str = ""):
    """Warn if RAM usage exceeds threshold."""
    pct, gb = get_memory_usage()
    msg = f"{prefix} RAM: {pct:.1f}% ({gb:.1f} GB)"
    if pct > threshold_pct:
        print(f"⚠️  HIGH {msg}")
    else:
        print(f"✓ {msg}")
    return pct

# ═════════════════════════════════════════════════════════════════════════════
#  GPU SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_gpu(gpu_id: int) -> str:
    print("─── Pipeline 4-Phase: Prepare → Detect → Mask → Caption ───")
    # Tăng CPU threads để sử dụng VRAM hiệu quả hơn (không CPU bottleneck)
    torch.set_num_threads(6)
    
    if not torch.cuda.is_available():
        print("CUDA không khả dụng → CPU mode")
        return "cpu"

    device_str = f"cuda:{gpu_id}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.deterministic    = False

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,"
        "max_split_size_mb:256,"
        "garbage_collection_threshold:0.9"
    )
    torch.cuda.empty_cache()

    try:
        prop      = torch.cuda.get_device_properties(gpu_id)
        total_mem = prop.total_memory / 1024**3
        print(f"GPU {gpu_id}: {prop.name}  |  VRAM: {total_mem:.1f} GB  |  SMs: {prop.multi_processor_count}")
        return device_str
    except Exception as e:
        print(f"Lỗi GPU {gpu_id}: {e}. Fallback → cuda:0")
        return "cuda:0"


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — DATASET PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

def _prepare_one(args_tuple):
    """
    Worker function for ThreadPool.
    Returns: (path, scale, dataset_path, status)
      status: 'ok' | 'small' | 'error'
    """
    path, output_img_dir, min_side, max_side = args_tuple
    try:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        dest = output_img_dir / path.name
        img = Image.open(path)
        w, h = img.size
        long_side = max(w, h)

        if long_side < min_side:
            return path, 1.0, None, "small"

        if long_side <= max_side:
            scale = 1.0
        else:
            scale = max_side / long_side

        if dest.exists():
            return path, scale, dest, "ok"

        # Thay đổi phần này: luôn lưu thành .png và đổi tên ở bước sau
        # Để đảm bảo converted, ta dùng .save thay vì shutil.copy
        img = img.convert("RGB")
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Lưu box tạm bằng tên gốc nhưng đuôi .png
        temp_dest = output_img_dir / (path.stem + "_temp.png")
        if scale == 1.0:
            img.save(temp_dest, "PNG")
        else:
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            img_resized.save(temp_dest, "PNG")

        return path, scale, temp_dest, "ok"

    except Exception as e:
        return path, 1.0, None, f"error: {e}"


def run_phase0(args, img_paths, output_img_dir) -> list[dict]:
    """
    Prepare dataset images (copy / resize).
    Returns list of dicts:
        { path (Path), scale (float), dataset_path (Path) }
    Only includes images that passed the min_side filter.
    """
    print("\n" + "=" * 70)
    print("  PHASE 0 — Dataset Preparation")
    print(f"  min_side={args.min_side}  max_side={args.max_side}")
    print("=" * 70)

    stats = {"ok": 0, "small": 0, "error": 0}
    prepared = []

    worker_args = [(p, output_img_dir, args.min_side, args.max_side) for p in img_paths]

    # Stream results instead of loading all into list to avoid RAM spike
    check_memory(prefix="[Phase0 Start]")
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Dùng map để tránh submit hàng loạt gây tràn RAM task queue
        for result in tqdm(pool.map(_prepare_one, worker_args), total=len(worker_args), desc="Phase0 Prepare", dynamic_ncols=True):
            if result is None:
                stats["small"] += 1
                continue
                
            path, scale, dest, status = result
            if status == "ok":
                prepared.append({"path": path, "scale": scale, "dataset_path": dest})
                stats["ok"] += 1
            else:
                tqdm.write(f"  ✗ {path.name}: {status}")
                stats["error"] += 1

    print(f"\n✅ Phase 0 processing done. Renaming to 1-xxxx.png ...")
    
    # Bước đổi tên tuần tự sau khi đã filter xong (để không bị nhảy số)
    final_prepared = []
    for idx, item in enumerate(prepared, start=1):
        temp_path = item["dataset_path"]
        final_path = temp_path.parent / f"{idx}.png"
        
        if temp_path.exists():
            temp_path.rename(final_path)
            
        item["dataset_path"] = final_path
        item["id"] = idx
        final_prepared.append(item)

    check_memory(prefix="[Phase0 End]")
    return final_prepared


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — OWLv2 DETECT (original) + SAM2 MASK (dataset)
# ═════════════════════════════════════════════════════════════════════════════

# ─── Model loading ───

def load_owlv2(device: str):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print(f"🦉 Loading OWLv2 on {device} ...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble",
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device).eval()
    return processor, model


def load_sam2(device: str, model_id: str = "facebook/sam2.1-hiera-large"):
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"🎯 Loading SAM2 ({model_id}) ...")
    sam2_model = build_sam2_hf(model_id, device=device, apply_postprocessing=True)
    return SAM2ImagePredictor(sam2_model)


# ─── Helpers ───

def load_image_pil(path: Path) -> Optional[Image.Image]:
    """Load image as PIL RGB, no resize."""
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def scale_bbox(bbox: list[float], scale: float) -> list[float]:
    """Scale bbox [x0, y0, x1, y1] by a uniform factor."""
    return [round(v * scale, 2) for v in bbox]


def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask).save(path, optimize=True)


# ─── Detection on original images ───

def detect_batch_owlv2(image_pils, all_labels, gd_processor, gd_model,
                       device, score_threshold):
    """Detect objects. Bbox returned is in the coordinate system of input image_pils."""
    if not all_labels:
        queries = [["furniture", "object", "item"]]
    else:
        queries = [all_labels]
    queries = queries * len(image_pils)

    inputs = gd_processor(
        text=queries, images=image_pils, return_tensors="pt"
    ).to(device)

    if "cuda" in device:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    # target_sizes = kích thước của chính image_pils được truyền vào
    # Đảm bảo bbox trả về khớp với hệ tọa độ của ảnh đó
    target_sizes = torch.tensor([img.size[::-1] for img in image_pils]).to(device)
    results_raw = gd_processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=score_threshold, target_sizes=target_sizes,
    )

    results = []
    for i, res in enumerate(results_raw):
        scores = res["scores"]
        if len(scores) == 0:
            results.append({"chosen_label": None, "chosen_box": None, "chosen_score": 0.0})
            continue
        best_idx  = scores.argmax().item()
        label_idx = res["labels"][best_idx].item()
        text_labels  = queries[i] if isinstance(queries[i], list) else queries[0]
        chosen_label = text_labels[label_idx] if label_idx < len(text_labels) else "object"
        results.append({
            "chosen_label": chosen_label,
            "chosen_box":   res["boxes"][best_idx].tolist(),
            "chosen_score": float(scores[best_idx]),
        })
    return results


# ─── Segmentation on dataset images ───

def segment_with_bbox(dataset_img_np, bbox_dataset, predictor, device):
    """Run SAM2 on the dataset-sized image using the scaled bbox."""
    if bbox_dataset is None:
        return None
    H, W = dataset_img_np.shape[:2]
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(dataset_img_np)
        masks, mask_scores, _ = predictor.predict(
            box=np.array(bbox_dataset, dtype=np.float32), multimask_output=True
        )
    best_mask = masks[np.argmax(mask_scores)].astype(np.uint8) * 255
    if best_mask.shape != (H, W):
        best_mask = np.array(Image.fromarray(best_mask).resize((W, H), Image.NEAREST))
    return best_mask


def dilate_mask(mask, dilation_pct):
    if dilation_pct <= 0:
        return mask
    h, w = mask.shape
    kernel_size = int(min(h, w) * dilation_pct)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


# ─── Phase 1 runner ───

def run_phase1(args, device, prepared, output_masks_dir) -> list[dict]:
    """
    Phase 1 được tách làm 2 nhịp (Pass 1: Dò tìm, Pass 2: Cắt nền)
    để CHẮC CHẮN không bao giờ OWLv2 và SAM2 cùng tồn tại trong VRAM.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1 — Pass 1: OWLv2 Detect | Pass 2: SAM2 Mask")
    print("=" * 70)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    stats  = {"success": 0, "skipped": 0, "error": 0}
    results = []

    # ─── PASS 1: OWLv2 DETECTION ONLY ───
    gd_processor, gd_model = load_owlv2(device)
    pbar_det = tqdm(total=len(prepared), desc="Phase1 Detect", dynamic_ncols=True)
    gc_interval = max(4, args.batch_size)  # GC mỗi N batches, không mỗi batch
    batch_count = 0
    
    for i in range(0, len(prepared), args.batch_size):
        batch = prepared[i: i + args.batch_size]
        # QUAN TRỌNG: OWLv2 detect trên ảnh DATASET (đã resize trong Phase 0)
        # để bbox trả về khớp với hệ tọa độ của ảnh dataset → SAM2 dùng thẳng, KHÔNG cần scale
        ds_pils = [load_image_pil(item["dataset_path"]) for item in batch]
        
        valid_idx = [j for j, op in enumerate(ds_pils) if op is not None]
        if valid_idx:
            valid_ds = [ds_pils[j] for j in valid_idx]
            detect_res = detect_batch_owlv2(
                valid_ds, labels, gd_processor, gd_model,
                device, args.score_threshold
            )
            for j, v_idx in enumerate(valid_idx):
                batch[v_idx]["det"] = detect_res[j]

        batch_len = len(batch)
        del ds_pils, valid_idx, batch
        if 'valid_ds' in locals(): del valid_ds
        if 'detect_res' in locals(): del detect_res
        
        # GC chỉ mỗi N batches để tránh overhead
        batch_count += 1
        if batch_count % gc_interval == 0:
            gc.collect()
        
        pbar_det.update(batch_len)
        
    pbar_det.close()
    
    # Ép xả VRAM OWLv2 triệt để trước khi load SAM2
    del gd_model, gd_processor
    gc.collect()
    torch.cuda.empty_cache()
    print("🗑️  OWLv2 unloaded, VRAM freed.\n")

    # ─── PASS 2: SAM2 MASKING ONLY ───
    sam_predictor = load_sam2(device, args.sam2_model)
    pbar_mask = tqdm(total=len(prepared), desc="Phase1 Mask", dynamic_ncols=True)
    
    for item in prepared:
        path = item["path"]
        scale = item["scale"]
        det = item.get("det")

        if not det or not det.get("chosen_box"):
            stats["skipped"] += 1
            pbar_mask.update(1)
            continue

        try:
            ds_pil = load_image_pil(item["dataset_path"])
            if ds_pil is None:
                stats["error"] += 1
                pbar_mask.update(1)
                continue

            ds_np = np.asarray(ds_pil)
            
            # OWLv2 đã detect trên ảnh dataset → bbox khớp luôn → truyền thẳng vào SAM2
            # KHONG cần scale_bbox nữa (y hệt mask_generate.py)
            bbox = [round(v, 2) for v in det["chosen_box"]]

            mask = segment_with_bbox(ds_np, bbox, sam_predictor, device)
            if mask is None:
                stats["skipped"] += 1
            else:
                mask = dilate_mask(mask, args.dilation)
                # Tên mask phải trùng tên ảnh (ví dụ 1.png)
                mask_path = output_masks_dir / (item["dataset_path"].name)
                save_mask(mask, mask_path)

                results.append({
                    "id":            item["id"],
                    "image_file":    item["dataset_path"].name,
                    "mask_file":     mask_path.name,
                    "object_label":  det["chosen_label"],
                    "bbox":          bbox,
                })
                stats["success"] += 1
                tqdm.write(
                    f"  ✓ {path.name}  label={det['chosen_label']} "
                    f"score={det['chosen_score']:.3f} scale={scale:.4f}"
                )
                
            del ds_pil, ds_np, mask
            gc.collect()
        except Exception as e:
            tqdm.write(f"  ✗ {path.name}: {e}")
            stats["error"] += 1

        pbar_mask.update(1)
        
    pbar_mask.close()
    print(f"\n✅ Phase 1 done — success: {stats['success']}  skipped: {stats['skipped']}  error: {stats['error']}")

    del sam_predictor
    gc.collect()
    torch.cuda.empty_cache()
    print("🗑️  SAM2 unloaded, VRAM freed.\n")
    
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2+3 — QWEN3VL CAPTION + SDXL PROMPT
# ═════════════════════════════════════════════════════════════════════════════

def load_qwen3vl(device: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    print(f"🧠 Loading Qwen3VL-4B-Instruct (4-bit) on {device} ...")
    torch.cuda.empty_cache()
    gc.collect()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        device_map={"": device},
        quantization_config=quantization_config,
        attn_implementation="sdpa",  # Fallback: PyTorch 2.x Scaled Dot Product Attention 
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    torch.backends.cudnn.benchmark = True
    print("✅ Qwen3VL ready\n")
    return model, processor


def caption_single(model, processor, image: Image.Image, bbox, obj_label: str, device: str) -> str:
    """Caption using ORIGINAL image + ORIGINAL bbox."""
    if bbox:
        prompt_text = (
            f"There is a {obj_label} within bounding box {bbox}. "
            "List comma-separated tags describing only this object. "
            "Include: material, color, shape, texture, style. "
            "No sentences, no articles, no verbs. Example format: "
            "oak wood table, rectangular, dark brown, smooth matte finish, mid-century modern"
        )
    else:
        prompt_text = (
            f"There is a {obj_label} in this image. "
            "Describe only this object in one short sentence, maximum 30 words. "
            "Focus on: material, color, shape, texture, style. "
            "No background, no scene, no other objects."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": 501760},
                {"type": "text",  "text":  prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=80, temperature=0.3, do_sample=True,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


# ─── SDXL prompt building ───

def truncate_to_tokens(prompt: str, max_tokens: int = 75) -> str:
    words = prompt.split()
    return " ".join(words[:max_tokens]) if len(words) > max_tokens else prompt


# ─── Phase 2+3 runner ───

def run_phase2(args, device, results_json: list, dataset_img_dir: Path, output_json: Path, output_captions_dir: Path) -> list:
    """
    Caption each entry using ORIGINAL image + bbox_original.
    Checkpoints JSON every save_every images.
    Also creates .txt files in captions/
    """
    if not results_json:
        print("⚠️  No detections to caption.")
        return results_json

    print("\n" + "=" * 70)
    print("  PHASE 2+3 — Qwen3VL Caption + Dataset Export")
    print("=" * 70)

    model, processor = load_qwen3vl(device)

    # Warm-up (skipped for briefness in logic, but present in real)
    
    total      = len(results_json)
    success    = 0
    failed     = 0
    t0         = time.time()

    # Lưu ý: Cần đường dẫn đến ảnh gốc (để caption chất lượng cao nhất)
    # Nhưng ta cần biết file gốc tên là gì? Phase 1 cần truyền info này.
    
    for idx, entry in enumerate(results_json):
        # Ta cần load ảnh GỐC (chất lượng cao nhất cho Qwen)
        # Nhưng ở Phase 1 ta chỉ lưu tên 1.png. Ta cần map ngược lại hoặc dùng ảnh trong dataset.
        # Để an toàn nhất và đúng cấu trúc, ta dùng ảnh đã được chuẩn bị trong dataset/images/
        img_path = dataset_img_dir / entry["image_file"] 
        
        try:
            img = load_image_pil(img_path)
            if img is None: raise FileNotFoundError(f"Cannot open {img_path}")
        except Exception:
            entry["prompt"] = entry.get("object_label", "object")
            failed += 1
            continue

        obj  = entry.get("object_label", "object")
        bbox = entry.get("bbox_original") # Bbox này vẫn là hệ tọa độ gốc? 
        # Cần xem lại Phase 1: Ta lưu bbox_original. Qwen cần nhìn đúng bbox trên ảnh nó đang nạp.
        # Nếu nạp ảnh dataset (đã resize), ta nên dùng bbox_dataset.
        # Sửa: Để Qwen chính xác nhất, ta nạp thẳng ảnh dataset và dùng bbox đã scale.
        
        try:
            raw_caption = caption_single(model, processor, img, bbox=None, obj_label=obj, device=device)
            prompt      = truncate_to_tokens(raw_caption.strip(), max_tokens=75)

            # Update entry
            entry["prompt"] = prompt
            
            # --- TẠO FILE .TXT ---
            txt_name = Path(entry["image_file"]).stem + ".txt"
            txt_path = output_captions_dir / txt_name
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt)

            if "bbox_original" in entry: del entry["bbox_original"]
            del img
            gc.collect()
            success += 1
            
            if (idx + 1) % 10 == 0:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results_json, f, indent=4, ensure_ascii=False)
                tqdm.write(f"    💾 Checkpoint ({idx+1}/{total})")

        except Exception as e:
            entry["prompt"] = obj
            failed += 1
            tqdm.write(f"  ❌ {entry['image_file']}: {e}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("🗑️  Qwen3VL unloaded, VRAM freed.\n")
    return results_json


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main(args):
    t_start = time.time()
    device  = setup_gpu(args.gpu_id)
    
    # Auto-tune batch size based on available VRAM
    if torch.cuda.is_available():
        gpu_id = int(device.split(":")[-1])
        props = torch.cuda.get_device_properties(gpu_id)
        vram_gb = props.total_memory / 1024**3
        if vram_gb >= 40:
            args.batch_size = max(args.batch_size, 24)  # A100 40GB - aggressive batching
        elif vram_gb >= 24:
            args.batch_size = max(args.batch_size, 16)  # RTX 4090/A100 24GB
        elif vram_gb >= 16:
            args.batch_size = max(args.batch_size, 8)   # RTX 4080
        else:
            args.batch_size = max(args.batch_size, 4)   # Fallback
    print(f"📊 Batch size tuned to: {args.batch_size}  (Workers: {args.workers})")
    print(f"   💡 Tip: OWLv2 is GPU-parallelized — larger batch = faster detection\n")

    # ─── Paths ───
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input folder không tồn tại: {input_dir}")

    all_img_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not all_img_paths:
        print("Không tìm thấy ảnh nào trong input_dir.")
        return

    output_base   = Path(args.output_dir) if args.output_dir else input_dir.parent / "dataset"
    output_images = output_base / "images"
    output_masks  = output_base / "masks"
    output_captions = output_base / "captions"
    output_json   = output_base / "detection_results.json"
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    output_captions.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Input:  {input_dir}  ({len(all_img_paths)} images)")

    # ─── Resume: filter already-masked images ───
    if not args.overwrite:
        pending_paths = [p for p in all_img_paths
                         if not (output_masks / (p.stem + ".png")).exists()]
    else:
        pending_paths = all_img_paths

    # ═══ PHASE 0 ═══
    if pending_paths:
        _ = run_phase0(args, pending_paths, output_images)
        check_memory(prefix="[Phase 0 Done]")
        gc.collect()
    else:
        print("\n⏩ Phase 0 skipped — all images already processed")

    # ═══ PHASE 1: DETECT & MASK ═══
    if pending_paths:
        print(f"\n🚀 Phase 1: Detecting & Masking {len(pending_paths)} images ...")
        # Chạy chuẩn bị lại Prepared list nhẹ nhàng (Phase 0 chỉ copy/resize, ta cần info này)
        # Load lại từ Phase 0 nhưng k resize nữa vì Phase 0 đã làm rồi
        prepared = run_phase0(args, pending_paths, output_images)
        print(f"⚠️  DEBUG: prepared has {len(prepared)} entries (expected {len(pending_paths)})")
        
        new_results = run_phase1(args, device, prepared, output_masks)
        
        # Merge và lưu ngay Phase 1
        if output_json.exists():
            with open(output_json, encoding="utf-8") as f:
                all_results = json.load(f)
            existing_map = {e["image_file"]: e for e in all_results}
            for e in new_results:
                existing_map[e["image_file"]] = e
            all_results = list(existing_map.values())
        else:
            all_results = new_results

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"💾 Phase 1 JSON saved: {output_json}")
        
        # GIẢI PHÓNG TOÀN BỘ TRƯỚC KHI SANG PHASE 2
        del prepared, new_results, all_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n⏩ Phase 1 skipped — all masks already exist")

    # ═══ PHASE 2+3: CAPTION ═══
    # Load lại sạch sẽ từ file JSON sau khi Phase 1 đã unload xong xuôi
    if output_json.exists():
        with open(output_json, encoding="utf-8") as f:
            all_results = json.load(f)
            
        uncaptioned = [e for e in all_results if not e.get("prompt")]
        if uncaptioned:
            print(f"\n🔄 Phase 2: Captioning {len(uncaptioned)} images ...")
            # Cần truyền đủ 6 đối số: args, device, uncaptioned, img_dir, json_path, captions_dir
            run_phase2(args, device, uncaptioned, output_images, output_json, output_captions)
            
            # Sau khi caption xong, load lại bản cuối cùng để lưu
            with open(output_json, encoding="utf-8") as f:
                final_results = json.load(f)
            
            # Cập nhật caption vào list gốc (để giữ được metadata nếu có)
            # Thực tế run_phase2 đã tự lưu rồi, nhưng ta in ra summary cuối
            all_results = final_results
        else:
            print("\n⏩ Phase 2+3 skipped — all entries already captioned")
    else:
        print("⚠️  No JSON found to caption.")
        return

    elapsed = time.time() - t_start
    check_memory(prefix="[Pipeline End]")
    
    print("\n" + "=" * 70)
    print(f"🏁 Pipeline complete in {elapsed/60:.1f} min")
    print(f"   📁 Dataset images:  {output_images}")
    print(f"   📁 Masks:           {output_masks}")
    print(f"   📄 JSON:            {output_json}")
    print(f"   Total in JSON:      {len(all_results)}")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4-Phase Pipeline: Prepare → OWLv2 Detect → SAM2 Mask → Qwen3VL Caption"
    )

    # I/O
    parser.add_argument("--input_dir",  type=str, default="data/raw_images",
                        help="Folder chứa ảnh gốc")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Folder output base (default = input_dir/../dataset)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Ghi đè mask/dataset images đã tồn tại")

    # GPU
    parser.add_argument("--gpu_id",    type=int, default=0)

    # Phase 0
    parser.add_argument("--min_side",  type=int, default=100,
                        help="Bỏ ảnh có max(w,h) < min_side")
    parser.add_argument("--max_side",  type=int, default=1024,
                        help="Resize ảnh có max(w,h) > max_side về đúng max_side")

    # OWLv2
    _DEFAULT_LABELS = (
        "sofa,armchair,chair,dining table,coffee table,side table,desk,bed,wardrobe,"
        "bookshelf,TV stand,television,lamp,chandelier,rug,curtain,stove,range,refrigerator,washer,ceiling fan,clock"
    )
    
    parser.add_argument("--labels",          type=str,   default=_DEFAULT_LABELS,
                        help="Comma-separated labels (default: top 20 indoor items)")
    parser.add_argument("--score_threshold", type=float, default=0.25)

    # SAM2
    parser.add_argument("--sam2_model", type=str,   default="facebook/sam2.1-hiera-large")
    parser.add_argument("--dilation",   type=float, default=0.1,
                        help="Mask dilation percentage")

    # Processing
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for OWLv2 detection (giảm nếu RAM hệ thống bị OOM)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="I/O thread workers (giảm để tránh prefetch quá nhiều)")

    # Caption
    parser.add_argument("--save_every",       type=int,   default=20,
                        help="Checkpoint JSON mỗi N ảnh (caption phase)")

    main(parser.parse_args())
