#!/usr/bin/env python
# coding=utf-8
"""
generate_masks_batch.py
───────────────────────
Xử lý cả folder ảnh, mỗi ảnh tạo 1 mask duy nhất từ 1 vật thể rõ nhất.
Tích hợp Florence-2 để tự động tạo caption chi tiết cho vật thể.
Tối ưu cho A100 / GPU cao cấp.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import cv2
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# GPU setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu() -> str:
    if not torch.cuda.is_available():
        print("CUDA không khả dụng → CPU mode (sẽ rất chậm)")
        return "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,"
        "max_split_size_mb:512,"
        "garbage_collection_threshold:0.8"
    )

    torch.cuda.empty_cache()
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU : {prop.name} | VRAM: {prop.total_memory / 1024**3:.1f} GB")
    return "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

def load_grounding_dino(device: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-base"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()
    return processor, model


def load_sam2_predictor(device: str, model_id: str = "facebook/sam2.1-hiera-large"):
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"Loading SAM2 ({model_id}) ...")
    sam2_model = build_sam2_hf(model_id, device=device, apply_postprocessing=True)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def load_caption_model(device: str):
    from transformers import AutoProcessor, AutoModelForCausalLM
    model_id = "microsoft/Florence-2-base"
    print(f"Loading Florence-2 for captioning...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return processor, model


def get_detailed_caption(processor, model, image_pil, box, device):
    """Cắt vùng object và tạo caption chi tiết bằng Florence-2."""
    # box: [x1, y1, x2, y2]
    # Mở rộng nhẹ vùng crop để model nhìn thấy ngữ cảnh
    w, h = image_pil.size
    x1, y1, x2, y2 = box
    pad_w = (x2 - x1) * 0.05
    pad_h = (y2 - y1) * 0.05
    crop_box = (max(0, x1-pad_w), max(0, y1-pad_h), min(w, x2+pad_w), min(h, y2+pad_h))
    
    crop = image_pil.crop(crop_box)
    
    prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=crop, return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

def detect_batch(image_pils, all_labels, gd_processor, gd_model, device, score_threshold):
    if not all_labels:
        text_prompt = "furniture . object . home items . item ."
    else:
        text_prompt = " . ".join(all_labels) + " ."

    inputs = gd_processor(
        images=image_pils, text=[text_prompt] * len(image_pils), return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    raw_results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[img.size[::-1] for img in image_pils]
    )

    results = []
    for res in raw_results:
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"]

        if len(scores) == 0:
            results.append({"chosen_label": None, "chosen_box": None, "chosen_score": 0.0})
            continue

        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])

        if best_score < score_threshold:
            results.append({"chosen_label": None, "chosen_box": None, "chosen_score": 0.0})
        else:
            results.append({
                "chosen_label": labels[best_idx],
                "chosen_box": boxes[best_idx].tolist(),
                "chosen_score": best_score,
            })
    return results


def segment_best_box(image_np, box, predictor, device):
    if box is None: return None
    H, W = image_np.shape[:2]
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(image_np)
        masks, mask_scores, _ = predictor.predict(box=np.array(box), multimask_output=True)
    best_mask = masks[np.argmax(mask_scores)].astype(np.uint8) * 255
    if best_mask.shape != (H, W):
        best_mask = np.array(Image.fromarray(best_mask).resize((W, H), Image.NEAREST))
    return best_mask


def dilate_mask(mask, dilation_pct):
    if dilation_pct <= 0: return mask
    h, w = mask.shape
    kernel_size = int(min(h, w) * dilation_pct)
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 1: return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def load_image(path: Path, max_side: int = 1024):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img, np.array(img)
    except: return None

def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask).save(path, optimize=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    t_start = time.time()
    device  = setup_gpu()

    input_dir   = Path(args.input_dir) if args.input_dir else None
    input_image = Path(args.input_image) if args.input_image else None

    if input_image:
        img_paths = [input_image]
        output_dir = input_image.parent / (input_image.parent.name + "_masks")
    elif input_dir:
        img_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        output_dir = input_dir.parent / (input_dir.name + "_masks")
    else: raise ValueError("Cần --input_dir hoặc --input_image")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        img_paths = [p for p in img_paths if not (output_dir / (p.stem + ".png")).exists()]
    
    if not img_paths:
        print("Không có ảnh cần xử lý.")
        return

    # Load models
    gd_processor, gd_model = load_grounding_dino(device)
    sam_predictor          = load_sam2_predictor(device, args.sam2_model)
    vlm_processor, vlm_model = load_caption_model(device)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    stats = {"success": 0, "skipped": 0, "error": 0}
    results_json = []

    io_pool = ThreadPoolExecutor(max_workers=args.workers)
    pbar = tqdm(total=len(img_paths), desc="Processing", dynamic_ncols=True)

    for i in range(0, len(img_paths), args.batch_size):
        batch_paths = img_paths[i: i + args.batch_size]
        load_results = [io_pool.submit(load_image, p, args.max_side).result() for p in batch_paths]
        
        valid_batch = []
        for p, res in zip(batch_paths, load_results):
            if res: valid_batch.append((p, res[0], res[1]))
            else: stats["error"] += 1

        if not valid_batch: continue

        batch_pils = [x[1] for x in valid_batch]
        detect_res = detect_batch(batch_pils, labels, gd_processor, gd_model, device, args.score_threshold)

        for (path, img_pil, img_np), det in zip(valid_batch, detect_res):
            if not det["chosen_box"]:
                stats["skipped"] += 1
                continue

            try:
                # 1. Segment
                mask = segment_best_box(img_np, det["chosen_box"], sam_predictor, device)
                if mask is None:
                    stats["skipped"] += 1
                    continue

                # 2. Caption
                caption = get_detailed_caption(vlm_processor, vlm_model, img_pil, det["chosen_box"], device)

                # 3. Dilation
                mask = dilate_mask(mask, args.dilation)

                # 4. Save
                mask_path = output_dir / (path.stem + ".png")
                io_pool.submit(save_mask, mask, mask_path)

                results_json.append({
                    "image_file": path.name,
                    "mask_file": mask_path.name,
                    "object_label": det["chosen_label"],
                    "caption": caption,
                    "confidence_score": round(det["chosen_score"], 4)
                })
                stats["success"] += 1
                tqdm.write(f"  ✓ {path.name}: {caption}")

            except Exception as e:
                tqdm.write(f"  ✗ {path.name}: {e}")
                stats["error"] += 1

        pbar.update(len(batch_paths))

    pbar.close()
    
    with open(output_dir / "detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"\nHoàn tất! Kết quả lưu tại {output_dir}")
    print(f"Thành công: {stats['success']} | Bỏ qua: {stats['skipped']} | Lỗi: {stats['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--input_image", type=str)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-large")
    parser.add_argument("--score_threshold", type=float, default=0.25)
    parser.add_argument("--dilation", type=float, default=0.20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_side", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())