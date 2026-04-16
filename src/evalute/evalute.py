#!/usr/bin/env python3
"""
Evaluation Script for SDXL Inpainting Results
Computes PSNR, SSIM, LPIPS, and FID metrics for 50 inpainting samples
"""

import os
import sys

# ============ MEMORY OPTIMIZATION (BEFORE importing torch) ============
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force GPU 1 only
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import re
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# Image processing imports
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)
from diffusers.utils import load_image
import torchvision.transforms as T
from transformers import DPTImageProcessor, DPTForDepthEstimation, CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

# Metrics imports
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not installed. Install with: pip install lpips")

from skimage.metrics import structural_similarity as ssim_metric

# SSIM threshold: >= này coi như model KHÔNG vẽ
SSIM_NOT_PAINTED_THRESHOLD = 0.50

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/diffusion/SDXL_Lora/src/evalute")
IMAGES_DIR = DATA_ROOT / "images_resized"
MASKS_DIR = DATA_ROOT / "masks_resized"
OUTPUT_DIR = DATA_ROOT / "results_sdxl"
NUM_SAMPLES = 50

# Memory optimization flags
COMPUTE_LPIPS = True   # Keep LPIPS metric

# Prompt enhancement
USE_QWEN_ENHANCE = False          # Bật/tắt Qwen prompt enhancement
QWEN_MODEL_ID    = "Qwen/Qwen3-4B"  # Model ID hoặc local path

# Pipeline configuration
PRETRAINED = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_ID = "diffusers/controlnet-depth-sdxl-1.0"
# LORA_PATH = "/home/diffusion/SDXL_Lora/train-3/sdxl-inpaint-lora/pytorch_lora_weights.safetensors"

# ==================== SINGLE GPU 1 SETUP ====================
# (CUDA_VISIBLE_DEVICES="1" already set before torch import above)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DUAL-GPU SETUP ====================
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print(f"🖥️  Detected {NUM_GPUS} GPU(s)")
    
    if NUM_GPUS >= 2:
        DEVICE_INFERENCE = torch.device("cuda:1")  # GPU 1: Load + Inference + LPIPS
        DEVICE_METRICS = torch.device("cuda:2")    # Metrics on GPU 0
        print(f"   GPU 1 (cuda:1) → Inference + LPIPS")
        print(f"   GPU 0 (cuda:2) → Metrics")
    else:
        DEVICE_INFERENCE = torch.device("cuda:2")
        DEVICE_METRICS = torch.device("cuda:2")    # Fallback to same GPU
        print(f"   Single GPU → Inference + Metrics on cuda:2")
else:
    NUM_GPUS = 0
    DEVICE_INFERENCE = DEVICE
    DEVICE_METRICS = DEVICE

# Inpainting parameters
INPAINT_PARAMS = {
    "strength": 1.0,
    "num_inference_steps": 50,
    "guidance_scale": 12.0,
    "controlnet_conditioning_scale": 0.3,
}

# Metrics parameters
LPIPS_NET = "alex"  # 'alex', 'vgg', 'squeeze'

# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def clear_vram():
    """Giải phóng GPU VRAM"""
    print("🧹 Đang dọn dẹp VRAM...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    print("✅ VRAM đã được làm trống.")


def estimate_depth(pil_image: Image.Image, dp, dm, device) -> Image.Image:
    """Estimate depth via DPT-Hybrid (Transformers) → returns RGB PIL Image."""
    W, H = pil_image.size
    with torch.no_grad():
        inputs = dp(images=pil_image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(device, torch.float16) for k, v in inputs.items()}
        pred = dm(**inputs).predicted_depth
    dt = torch.nn.functional.interpolate(
        pred.unsqueeze(1).float(), size=(H, W), mode="bicubic", align_corners=False,
    ).squeeze()
    dn = dt.cpu().numpy()
    dn = (dn - dn.min()) / (dn.max() - dn.min() + 1e-8) * 255.0
    depth_rgb = np.stack([dn.astype(np.uint8)] * 3, axis=-1)
    return Image.fromarray(depth_rgb)


def regularize_mask(mask_image, target_size, roundness_threshold=0.75):
    """Regularize mask to be circular/elliptical shape."""
    mask_np = np.array(mask_image.convert("L").resize(target_size, Image.NEAREST))
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_image.convert("L").resize(target_size, Image.NEAREST)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0
    canvas = np.zeros_like(binary)
    if circularity >= roundness_threshold and len(contour) >= 5:
        (cx, cy), (ma, mb), angle = cv2.fitEllipse(contour)
        if min(ma, mb) / max(ma, mb) > 0.90:
            cv2.circle(canvas, (int(cx), int(cy)), int(max(ma, mb) / 2), 255, -1)
        else:
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(ma/2), int(mb/2)),
                        angle, 0, 360, 255, -1)
    else:
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
        cv2.fillPoly(canvas, [box], 255)
    return Image.fromarray(canvas)


def prepare_masked_image(image, mask, fill_color=(128, 128, 128)):
    """Tô xám vùng mask trong ảnh input → model không thấy nội dung cũ."""
    img_np = np.array(image.convert("RGB")).copy()
    mask_np = np.array(mask.convert("L"))
    img_np[mask_np > 127] = fill_color
    return Image.fromarray(img_np)


def prepare_masked_depth(depth, mask):
    """Tô phẳng depth trong vùng mask → ControlNet không bias hình dạng cũ."""
    d_np = np.array(depth.convert("RGB")).copy()
    m_np = np.array(mask.convert("L"))
    # Lấy median depth của vùng xung quanh mask làm giá trị phẳng
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    dilated = cv2.dilate(m_np, kernel)
    border = (dilated > 127) & (m_np <= 127)
    if border.any():
        median_val = int(np.median(d_np[border][:, 0]))
    else:
        median_val = 128
    d_np[m_np > 127] = median_val
    return Image.fromarray(d_np)


def dilate_mask(mask_pil, dilate_px=20):
    m = np.array(mask_pil)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
    dilated = cv2.dilate(m, kernel, iterations=1)
    return Image.fromarray(dilated)


def load_prompts(filepath, num_samples):
    """Load prompts from text file (one per line)."""
    prompts = {}
    if not Path(filepath).exists():
        print(f"⚠️  Prompt file not found: {filepath}")
        print(f"   Using default prompt for all samples")
        return {i: "a beautiful interior design object" for i in range(1, num_samples + 1)}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines[:num_samples], 1):
            prompt = line.strip()
            if prompt:
                prompts[idx] = prompt
            else:
                prompts[idx] = "a beautiful interior design object"
        
        # Fill missing with default
        for idx in range(1, num_samples + 1):
            if idx not in prompts:
                prompts[idx] = "a beautiful interior design object"
        
        print(f"✅ Loaded {len(prompts)} prompts from: {filepath}")
        return prompts
    
    except Exception as e:
        print(f"⚠️  Error reading prompt file: {e}")
        return {i: "a beautiful interior design object" for i in range(1, num_samples + 1)}


def inpaint_with_depth(pipe, init_image, mask_image, depth_image, prompt, negative_prompt,
                       inpaint_size=1024, crop_padding=128, **pipe_kwargs):
    """Crop mask region with adaptive padding, inpaint with depth conditioning, paste back."""
    W, H = init_image.size
    mask_np = np.array(mask_image.convert("L"))
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        raise ValueError("Mask is empty")

    # Adaptive padding: small mask → larger padding for more context
    mask_w = int(xs.max()) - int(xs.min())
    mask_h = int(ys.max()) - int(ys.min())
    adaptive_pad = int(max(crop_padding, min(max(mask_w, mask_h) * 0.8, W * 0.25)))

    x1 = max(0, int(xs.min()) - adaptive_pad)
    y1 = max(0, int(ys.min()) - adaptive_pad)
    x2 = min(W, int(xs.max()) + adaptive_pad)
    y2 = min(H, int(ys.max()) + adaptive_pad)

    print(f"  Ảnh gốc : {W}×{H}   Mask: {mask_w}×{mask_h}px")
    print(f"  Padding : {adaptive_pad}px (adaptive)   Crop: ({x1},{y1})→({x2},{y2}) [{x2-x1}×{y2-y1}]")

    crop_img   = init_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_mask  = mask_image.convert("L").crop((x1, y1, x2, y2))
    crop_depth = depth_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_w, crop_h = crop_img.size

    # Scale to inpaint_size (divisible by 64 for SDXL)
    scale = inpaint_size / max(crop_w, crop_h)
    inp_w = (int(crop_w * scale) // 64) * 64
    inp_h = (int(crop_h * scale) // 64) * 64

    mask_px_at_inp = int(max(mask_w, mask_h) * scale)
    print(f"  Inpaint : {inp_w}×{inp_h}   Mask tại inp: ~{mask_px_at_inp}px (latent: ~{mask_px_at_inp//8}px)")

    inp_img   = crop_img.resize((inp_w, inp_h), Image.LANCZOS)
    inp_mask  = crop_mask.resize((inp_w, inp_h), Image.NEAREST)
    inp_depth = crop_depth.resize((inp_w, inp_h), Image.LANCZOS)

    # Inpaint with depth conditioning
    result_small = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=inp_img,
        mask_image=inp_mask,
        control_image=inp_depth,
        height=inp_h,
        width=inp_w,
        **pipe_kwargs,
    ).images[0]

    # Paste back to original resolution
    result_crop = result_small.resize((crop_w, crop_h), Image.LANCZOS)
    output = init_image.convert("RGB").copy()
    output.paste(result_crop, (x1, y1), mask=crop_mask.resize((crop_w, crop_h), Image.NEAREST))

    return output


def alpha_blend(src, dst, mask, feather=5): # ← Hạ feather xuống 5
    mask_f = cv2.GaussianBlur(mask.astype(float), (feather*2+1, feather*2+1), 0) / 255.0
    mask_f = mask_f[:, :, np.newaxis]
    blended = (src.astype(float) * mask_f + dst.astype(float) * (1 - mask_f))
    return blended.astype(np.uint8)


def safe_poisson_blend(src, dst, mask, margin=3): # ← Hạ margin xuống 3
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    # Erode (thu nhỏ) mask vào trong vài pixel để lẩn tránh viền rác
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    H, W = mask.shape
    if (mask_eroded[0].max() > 0 or mask_eroded[-1].max() > 0 or
        mask_eroded[:, 0].max() > 0 or mask_eroded[:, -1].max() > 0):
        print("⚠️ Mask chạm biên → Chuyển sang alpha blend an toàn")
        return alpha_blend(src, dst, mask)
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return alpha_blend(src, dst, mask)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    center = (x + w // 2, y + h // 2)
    try:
        return cv2.seamlessClone(src, dst, mask_eroded, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print(f"⚠️ Poisson fail: {e} → Chuyển sang alpha blend an toàn")
        return alpha_blend(src, dst, mask)


def inpaint_mask_aware(
    pipe, init_image, mask_image, depth_map,
    prompt, negative_prompt,
    inpaint_size=1024,
    target_mask_fraction=0.5,
    min_mask_px=400,
    max_context_ratio=0.85,
    **pipe_kwargs,
):
    W, H = init_image.size
    mask_np = np.array(mask_image.convert("L"))
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        raise ValueError("Mask trống")

    mx1, mx2 = int(xs.min()), int(xs.max())
    my1, my2 = int(ys.min()), int(ys.max())
    mask_w = mx2 - mx1
    mask_h = my2 - my1
    cx = (mx1 + mx2) / 2
    cy = (my1 + my2) / 2

    # 1. Tính toán scale và crop vùng inpaint
    target_mask_px = max(min_mask_px, inpaint_size * target_mask_fraction)
    scale = target_mask_px / max(mask_w, mask_h)
    crop_size = inpaint_size / scale

    max_crop = min(W, H) * max_context_ratio
    if crop_size > max_crop:
        crop_size = max_crop
        scale = inpaint_size / crop_size

    half = crop_size / 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(W, cx + half))
    y2 = int(min(H, cy + half))

    # Adjust nếu bị clamp ở biên
    if x2 - x1 < crop_size:
        if x1 == 0: x2 = min(W, int(crop_size))
        else: x1 = max(0, int(x2 - crop_size))
    if y2 - y1 < crop_size:
        if y1 == 0: y2 = min(H, int(crop_size))
        else: y1 = max(0, int(y2 - crop_size))

    crop_w, crop_h = x2 - x1, y2 - y1
    inp_w = (int(crop_w * scale) // 64) * 64
    inp_h = (int(crop_h * scale) // 64) * 64

    # 2. Tạo các bản crop cho vùng inpaint (hard mask NEAREST)
    crop_img   = init_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_mask  = mask_image.convert("L").crop((x1, y1, x2, y2))
    crop_depth = depth_map.convert("RGB").crop((x1, y1, x2, y2))

    inp_img   = crop_img.resize((inp_w, inp_h), Image.LANCZOS)
    inp_mask  = crop_mask.resize((inp_w, inp_h), Image.NEAREST)
    inp_depth = crop_depth.resize((inp_w, inp_h), Image.LANCZOS)

    # 3. GLOBAL CONTEXT STRIP — model thấy toàn cảnh ảnh
    context_strip_w = 256  # phải chia hết cho 64
    context_thumb       = init_image.resize((context_strip_w, inp_h), Image.LANCZOS)
    context_depth_thumb = depth_map.resize((context_strip_w, inp_h), Image.LANCZOS)

    combined_w = inp_w + context_strip_w
    combined_inp = Image.new("RGB", (combined_w, inp_h))
    combined_inp.paste(context_thumb, (0, 0))
    combined_inp.paste(inp_img, (context_strip_w, 0))

    # Mask vùng context = 0 (đen) để model không vẽ đè
    combined_mask = Image.new("L", (combined_w, inp_h), 0)
    combined_mask.paste(inp_mask, (context_strip_w, 0))

    combined_depth = Image.new("RGB", (combined_w, inp_h))
    combined_depth.paste(context_depth_thumb, (0, 0))
    combined_depth.paste(inp_depth, (context_strip_w, 0))

    print(f"  Crop size    : {crop_w}x{crop_h}")
    print(f"  Inpaint Res  : {inp_w}x{inp_h}")
    print(f"  Combined Res : {combined_w}x{inp_h} (Context strip: {context_strip_w}px)")

    # 4. Chạy model Inpaint
    result_combined = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=combined_inp,
        mask_image=combined_mask,
        control_image=combined_depth,
        height=inp_h,
        width=combined_w,
        **pipe_kwargs,
    ).images[0]

    # 5. Tách kết quả, paste về ảnh gốc bằng hard mask NEAREST
    result_small = result_combined.crop((context_strip_w, 0, combined_w, inp_h))
    result_crop  = result_small.resize((crop_w, crop_h), Image.LANCZOS)
    output = init_image.convert("RGB").copy()
    output.paste(result_crop, (x1, y1),
                 mask=crop_mask.resize((crop_w, crop_h), Image.NEAREST))

    return output


def inpaint_mask_aware_feathered(
    pipe, init_image, mask_image, depth_map,
    prompt, negative_prompt,
    inpaint_size=1024,
    target_mask_fraction=0.5,
    min_mask_px=400,
    max_context_ratio=0.85,
    **pipe_kwargs,
):
    """Feathered inpaint: context strip 256px + soft mask blending (no Poisson)."""
    W, H = init_image.size

    # Bước 1: Hard threshold để tính crop (mask_image đã là soft mask)
    mask_np_hard = np.array(mask_image.convert("L"))
    ys, xs = np.where(mask_np_hard > 127)
    if len(xs) == 0:
        raise ValueError("Mask trống")

    mx1, mx2 = int(xs.min()), int(xs.max())
    my1, my2 = int(ys.min()), int(ys.max())
    mask_w = mx2 - mx1
    mask_h = my2 - my1
    cx = (mx1 + mx2) / 2
    cy = (my1 + my2) / 2

    # Bước 2: Tính scale và crop size
    target_mask_px = max(min_mask_px, inpaint_size * target_mask_fraction)
    scale = target_mask_px / max(mask_w, mask_h)
    crop_size = inpaint_size / scale

    max_crop = min(W, H) * max_context_ratio
    if crop_size > max_crop:
        crop_size = max_crop
        scale = inpaint_size / crop_size

    half = crop_size / 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(W, cx + half))
    y2 = int(min(H, cy + half))

    if x2 - x1 < crop_size:
        if x1 == 0: x2 = min(W, int(crop_size))
        else: x1 = max(0, int(x2 - crop_size))
    if y2 - y1 < crop_size:
        if y1 == 0: y2 = min(H, int(crop_size))
        else: y1 = max(0, int(y2 - crop_size))

    crop_w, crop_h = x2 - x1, y2 - y1
    inp_w = (int(crop_w * scale) // 64) * 64
    inp_h = (int(crop_h * scale) // 64) * 64

    # Bước 3: Crop dùng soft mask (LANCZOS để giữ độ mờ)
    crop_img       = init_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_mask_soft = mask_image.convert("L").crop((x1, y1, x2, y2))
    crop_depth     = depth_map.convert("RGB").crop((x1, y1, x2, y2))

    inp_img   = crop_img.resize((inp_w, inp_h), Image.LANCZOS)
    inp_mask  = crop_mask_soft.resize((inp_w, inp_h), Image.LANCZOS)
    inp_depth = crop_depth.resize((inp_w, inp_h), Image.LANCZOS)

    # Bước 4: Context strip 256px
    context_strip_w     = 256
    context_thumb       = init_image.resize((context_strip_w, inp_h), Image.LANCZOS)
    context_depth_thumb = depth_map.resize((context_strip_w, inp_h), Image.LANCZOS)

    combined_w = inp_w + context_strip_w
    combined_inp = Image.new("RGB", (combined_w, inp_h))
    combined_inp.paste(context_thumb, (0, 0))
    combined_inp.paste(inp_img, (context_strip_w, 0))

    combined_mask = Image.new("L", (combined_w, inp_h), 0)
    combined_mask.paste(inp_mask, (context_strip_w, 0))

    combined_depth = Image.new("RGB", (combined_w, inp_h))
    combined_depth.paste(context_depth_thumb, (0, 0))
    combined_depth.paste(inp_depth, (context_strip_w, 0))

    print(f"  Crop size    : {crop_w}x{crop_h}")
    print(f"  Inpaint Res  : {inp_w}x{inp_h}")
    print(f"  Combined Res : {combined_w}x{inp_h} (Context strip: {context_strip_w}px)")

    # Bước 5: Inpaint
    result_combined = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=combined_inp,
        mask_image=combined_mask,
        control_image=combined_depth,
        height=inp_h,
        width=combined_w,
        **pipe_kwargs,
    ).images[0]

    # Bước 6: Tách kết quả + feather blend bằng soft mask
    result_small = result_combined.crop((context_strip_w, 0, combined_w, inp_h))
    result_crop  = result_small.resize((crop_w, crop_h), Image.LANCZOS)
    output = init_image.convert("RGB").copy()
    output.paste(result_crop, (x1, y1), mask=crop_mask_soft)

    return output


def enhance_prompt(raw_prompt, pipe, qwen_tokenizer, qwen_model, device, max_clip_tokens=77):
    """Enhance a raw prompt using Qwen3-4B LLM."""
    system = (
        "You are an expert Stable Diffusion prompt writer for interior design inpainting. "
        "As an interior design expert, please help me enhance the simple descriptions "
        "of an object into a detailed, vivid presentation of that object."
        "Given a enchance description, rewrite it into a detailed, high-quality prompt. "
        "Rules:\n1. Output ONLY the enhanced prompt\n2. Under 60 words\n"
        "3. Add: material, lighting, style, quality tags\n4. Comma-separated\n5. No quotes\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Enhance this inpainting prompt: {raw_prompt}"},
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = qwen_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = qwen_model.generate(
            **inputs, max_new_tokens=120, temperature=0.7,
            top_p=0.9, do_sample=True,
        )
    enhanced = qwen_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    enhanced = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", enhanced, flags=re.DOTALL).strip()
    enhanced = re.sub(r"<\s*think\s*>.*", "", enhanced, flags=re.DOTALL).strip()
    enhanced = re.sub(r"<[^>]+>", "", enhanced).strip().strip('"\'')
    if not enhanced:
        return raw_prompt
    clip_ids = pipe.tokenizer.encode(enhanced)
    if len(clip_ids) > max_clip_tokens:
        enhanced = pipe.tokenizer.decode(
            clip_ids[:max_clip_tokens], skip_special_tokens=True
        )
    return enhanced


# ──────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────

def evaluate_inpaint(result, original, mask_image, prompt, lpips_fn, clip_model, clip_proc, device):
    """Compute inpainting metrics: LPIPS (background), Boundary Consistency, CLIP Score, Color Consistency."""
    mask_np = np.array(mask_image.convert("L"))
    r_np    = np.array(result.convert("RGB")).astype(float)
    o_np    = np.array(original.convert("RGB")).astype(float)
    to_t    = T.ToTensor()

    white_px = (mask_np > 127).sum()
    total_px = mask_np.size
    print(f"  Mask white pixels: {white_px} / {total_px} ({white_px/total_px*100:.2f}%)")
    if white_px == 0:
        print("  ❌ MASK RỖNG — kiểm tra lại file mask!")
        return None

    # ── 1. LPIPS ngoài mask ───────────────────────────────────────
    outside = (mask_np == 0).astype(np.uint8)
    r_t = to_t(result).unsqueeze(0).to(device) * 2 - 1
    o_t = to_t(original).unsqueeze(0).to(device) * 2 - 1
    outside_t = torch.from_numpy(outside).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_val = lpips_fn(
            r_t * outside_t.expand_as(r_t),
            o_t * outside_t.expand_as(o_t)
        ).item()

    # ── 2. Boundary Consistency ───────────────────────────────────
    kernel   = np.ones((11, 11), np.uint8)
    boundary = cv2.dilate(mask_np, kernel) - cv2.erode(mask_np, kernel)
    bc_score = np.abs(r_np - o_np)[boundary > 0].mean() if boundary.sum() > 0 else 0.0

    # ── 3. CLIP Score — crop đúng vùng mask ──────────────────────
    ys, xs = np.where(mask_np > 127)
    pad = 30
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(result.width,  int(xs.max()) + pad)
    y2 = min(result.height, int(ys.max()) + pad)
    inpaint_crop = result.crop((x1, y1, x2, y2))
    print(f"  CLIP crop        : ({x1},{y1})→({x2},{y2}) [{x2-x1}×{y2-y1}px]")

    inputs = clip_proc(text=[prompt], images=inpaint_crop, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        clip_score = clip_model(**inputs).logits_per_image.item()

    # ── 4. Color Consistency ──────────────────────────────────────
    context_kernel = np.ones((31, 31), np.uint8)
    context_region = cv2.dilate(mask_np, context_kernel) - mask_np
    if context_region.sum() > 0 and mask_np.sum() > 0:
        inpaint_color = r_np[mask_np > 127].mean(axis=0)
        context_color = r_np[context_region > 127].mean(axis=0)
        color_diff    = np.abs(inpaint_color - context_color).mean()
    else:
        color_diff = 0.0

    print(f"  LPIPS (nền)          : {lpips_val:.4f}  {'✅' if lpips_val < 0.05 else '⚠️' if lpips_val < 0.15 else '❌'}  (< 0.05 tốt)")
    print(f"  Boundary Consistency : {bc_score:.2f}    {'✅' if bc_score < 8 else '⚠️' if bc_score < 20 else '❌'}  (< 8 tốt)")
    print(f"  CLIP Score           : {clip_score:.2f}    {'✅' if clip_score > 28 else '⚠️' if clip_score > 22 else '❌'}  (> 28 tốt)")
    print(f"  Color Consistency    : {color_diff:.2f}    {'✅' if color_diff < 10 else '⚠️' if color_diff < 20 else '❌'}  (< 10 tốt)")

    return {
        "lpips":      round(lpips_val, 4),
        "boundary":   round(bc_score, 2),
        "clip_score": round(clip_score, 2),
        "color_diff": round(color_diff, 2),
    }


# ──────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SDXL Inpainting Evaluation Script (2-Phase)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print()
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inpaint_dir = OUTPUT_DIR / "inpainted_images"
    inpaint_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = OUTPUT_DIR / "originals"  # Save originals for metrics phase
    originals_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts_dict = load_prompts(str(DATA_ROOT / "prompts.txt"), NUM_SAMPLES)
    
    # Default negative prompt
    negative_prompt = (
        "additional objects, extra items, unwanted objects, "
        "multiple objects, cluttered, busy background, "
        "unrelated decoration, extra furniture, "
        "low quality, blurry, distorted, deformed, artifacts, "
        "3d render, CGI, plastic texture, flat shading, "
        "floating, wrong perspective, impossible geometry, "
        "cartoon, anime, painting, sketch, "
        "text, watermark, logo"
    )
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: INFERENCE ONLY (Load models → inpaint → save → unload)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 1: INFERENCE")
    print("=" * 80)
    
    clear_vram()
    print("📦 Loading inference models...")
    
    # Load SDXL pipeline
    print("  → Loading SDXL ControlNet pipeline...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        PRETRAINED,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)
    
    # Load LoRA if exists
    # try:
    #     pipe.load_lora_weights(LORA_PATH)
    #     pipe.set_adapters(["default_0"], adapter_weights=[0.8])
    #     print("  ✅ LoRA loaded")
    # except Exception as e:
    #     print(f"  ⚠️  LoRA not loaded: {e}")
    print("  ℹ️  LoRA disabled — evaluating base model")
    
    pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    # pipe.enable_attention_slicing()
    
    # Load DPT depth estimator
    print("  → Loading DPT depth estimator...")
    dp = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    dm = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas", torch_dtype=torch.float16
    ).to(DEVICE)
    dm.eval()
    print("  ✅ DPT depth estimator ready")

    # Load Qwen for prompt enhancement (optional)
    qwen_tokenizer = None
    qwen_model = None
    if USE_QWEN_ENHANCE:
        print("  → Loading Qwen prompt enhancer...")
        try:
            qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.float16,
                device_map={"":str(DEVICE)},
            )
            qwen_model.eval()
            print(f"  ✅ Qwen ({QWEN_MODEL_ID}) loaded")
        except Exception as e:
            print(f"  ⚠️  Qwen không load được: {e}")
            qwen_tokenizer = None
            qwen_model = None
    else:
        print("  ℹ️  Qwen enhancement disabled (USE_QWEN_ENHANCE=False)")

    torch.cuda.empty_cache()
    gc.collect()
    clear_vram()
    print("✅ Tất cả model đã nằm gọn trên GPU. Không còn trùng lặp!\n")
    
    # Track which samples succeeded
    successful_samples = []
    
    print("🚀 Running inference on all samples...")
    for idx in tqdm(range(1, NUM_SAMPLES + 1), desc="Phase 1 - Inference"):
        try:
            img_path = IMAGES_DIR / f"{idx}.jpg"
            mask_path = MASKS_DIR / f"{idx}.png"
            
            init_image = load_image(str(img_path)).convert("RGB")
            W, H = init_image.size
            mask_image = regularize_mask(load_image(str(mask_path)), target_size=(W, H))

            # Estimate depth
            with torch.no_grad():
                depth_image = estimate_depth(init_image, dp, dm, DEVICE)

            raw_prompt = prompts_dict.get(idx, "a beautiful interior design object")
            if USE_QWEN_ENHANCE and qwen_tokenizer is not None and qwen_model is not None:
                prompt = enhance_prompt(raw_prompt, pipe, qwen_tokenizer, qwen_model, DEVICE)
                print(f"  📝 Raw  : {raw_prompt}")
                print(f"  ✨ Enhanced: {prompt}")
            else:
                prompt = raw_prompt
                print(f"  📝 Prompt: {prompt}")

            # Inpaint — hard mask + context strip 256px (giống hệt inference.py)
            result_raw = inpaint_mask_aware(
                pipe, init_image, mask_image, depth_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                inpaint_size=1024,
                target_mask_fraction=0.5,
                min_mask_px=400,
                max_context_ratio=0.85,
                generator=torch.Generator(DEVICE).manual_seed(42),
                **INPAINT_PARAMS,
            )

            # Poisson Seamless Blend — giống hệt inference.py Run section
            print("🚀 Đang thực hiện Poisson Blending...")
            dst = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
            src = cv2.cvtColor(np.array(result_raw), cv2.COLOR_RGB2BGR)
            mask_blend = np.array(mask_image.convert("L"))
            contours, _ = cv2.findContours(mask_blend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                center = (x + w // 2, y + h // 2)
                try:
                    blended_cv = cv2.seamlessClone(src, dst, mask_blend, center, cv2.NORMAL_CLONE)
                    result = Image.fromarray(cv2.cvtColor(blended_cv, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    print(f"⚠️ Poisson Blend lỗi: {e}. Dùng ảnh inpaint gốc.")
                    result = result_raw
            else:
                result = result_raw

            # Save result + original to disk
            result_filename = inpaint_dir / f"{idx}_sdxl.jpg"
            result.save(result_filename)
            init_image.save(originals_dir / f"{idx}.jpg")
            successful_samples.append(idx)
            
            # Cleanup iteration variables
            del result_raw, result, depth_image, mask_image, init_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error sample {idx}: {e}")
            result_raw = result = depth_image = mask_image = init_image = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
    
    print(f"\n✅ Phase 1 complete: {len(successful_samples)}/{NUM_SAMPLES} samples inpainted")

    # Unload Qwen sau khi inference xong
    if qwen_model is not None:
        print("🧹 Unloading Qwen model...")
        del qwen_tokenizer, qwen_model
        qwen_tokenizer = qwen_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1.5: LỌcC ẢNH KHÔNG VẼ (SSIM filter)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 1.5: LỌC ẢNH KHÔNG VẼ (SSIM filter)")
    print("=" * 80)
    print(f"   Ngưỡng SSIM: >= {SSIM_NOT_PAINTED_THRESHOLD} → không vẽ")

    painted_samples = []
    not_painted_samples = []

    for idx in successful_samples:
        orig_path = originals_dir / f"{idx}.jpg"
        result_path = inpaint_dir / f"{idx}_sdxl.jpg"
        mask_path = MASKS_DIR / f"{idx}.png"

        if not result_path.exists() or not mask_path.exists():
            painted_samples.append(idx)
            continue

        original = Image.open(orig_path).convert("RGB")
        result = Image.open(result_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if mask.size != original.size:
            mask = mask.resize(original.size, Image.NEAREST)
        if result.size != original.size:
            result = result.resize(original.size, Image.LANCZOS)

        mask_np = np.array(mask)
        ys, xs = np.where(mask_np > 127)

        if len(xs) == 0:
            painted_samples.append(idx)
            continue

        # Tính pixel diff CHỈ trong vùng mask (không tính background)
        orig_np = np.array(original.convert("RGB")).astype(float)
        result_np = np.array(result.convert("RGB")).astype(float)
        mask_bool = mask_np > 127

        mean_diff = np.abs(orig_np[mask_bool] - result_np[mask_bool]).mean()

        # SSIM trên crop bbox (chỉ để log thêm)
        pad = 10
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(original.width, int(xs.max()) + pad)
        y2 = min(original.height, int(ys.max()) + pad)
        orig_crop = np.array(original.crop((x1, y1, x2, y2)))
        result_crop = np.array(result.crop((x1, y1, x2, y2)))
        ssim_val = ssim_metric(orig_crop, result_crop, channel_axis=2, data_range=255)

        # Phán xét: mean_diff < 8 mới là "không vẽ"
        NOT_PAINTED_DIFF_THRESHOLD = 8.0
        if mean_diff < NOT_PAINTED_DIFF_THRESHOLD:
            not_painted_samples.append(idx)
            print(f"  [{idx:02d}] MaskDiff={mean_diff:.1f}  SSIM={ssim_val:.4f}  ❌ KHÔNG VẼ → loại khỏi metrics")
        else:
            painted_samples.append(idx)
            print(f"  [{idx:02d}] MaskDiff={mean_diff:.1f}  SSIM={ssim_val:.4f}  ✅ CÓ VẼ")

        del original, result, mask

    print(f"\n📊 Kết quả lọc: {len(painted_samples)} có vẽ / {len(not_painted_samples)} không vẽ")
    if not_painted_samples:
        print(f"   ❌ Không vẽ (bỏ qua): {not_painted_samples}")

    # Chỉ đưa vào Phase 2 những sample thực sự được vẽ
    successful_samples = painted_samples
    
    # ==================== UNLOAD ALL INFERENCE MODELS ====================
    print("🧹 Unloading inference models...")
    del pipe, controlnet, dp, dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("✅ GPU memory freed!\n")
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: METRICS (Load from disk → compute PSNR/SSIM/LPIPS/FID)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PHASE 2: METRICS")
    print("=" * 80)
    
    # Use GPU 1 (cuda:1) for metrics to avoid conflict with other processes on GPU 0
    METRICS_GPU = torch.device("cuda:1") if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else DEVICE
    print(f"   Metrics GPU: {METRICS_GPU}")
    
    metrics_data = {"lpips": [], "boundary": [], "clip_score": [], "color_diff": []}

    # Load LPIPS model
    lpips_fn = None
    if LPIPS_AVAILABLE and COMPUTE_LPIPS:
        print("  → Loading LPIPS model...")
        lpips_fn = lpips.LPIPS(net=LPIPS_NET).to(METRICS_GPU)
        lpips_fn.eval()
        print(f"  ✅ LPIPS ready on {METRICS_GPU}")

    # Load CLIP model
    print("  → Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(METRICS_GPU)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print(f"  ✅ CLIP ready on {METRICS_GPU}")

    print("📊 Computing metrics from saved images...")
    for idx in tqdm(successful_samples, desc="Phase 2 - Metrics"):
        try:
            # Load images from disk
            original = Image.open(originals_dir / f"{idx}.jpg").convert("RGB")

            # Find result file
            result_path = inpaint_dir / f"{idx}_sdxl.jpg"
            if not result_path.exists():
                print(f"⚠️  Sample {idx}: No result found, skipping")
                continue

            result = Image.open(result_path).convert("RGB")

            # Load mask
            mask_path = MASKS_DIR / f"{idx}.png"
            if not mask_path.exists():
                print(f"⚠️  Sample {idx}: No mask found, skipping")
                continue
            mask_image = Image.open(mask_path).convert("L")
            if mask_image.size != original.size:
                mask_image = mask_image.resize(original.size, Image.NEAREST)
            if result.size != original.size:
                result = result.resize(original.size, Image.LANCZOS)

            # Get prompt
            prompt = prompts_dict.get(idx, "a beautiful interior design object")
            print(f"\n[Sample {idx}]")
            scores = evaluate_inpaint(
                result=result,
                original=original,
                mask_image=mask_image,
                prompt=prompt,
                lpips_fn=lpips_fn,
                clip_model=clip_model,
                clip_proc=clip_proc,
                device=METRICS_GPU,
            )
            if scores is not None:
                metrics_data["lpips"].append(scores["lpips"])
                metrics_data["boundary"].append(scores["boundary"])
                metrics_data["clip_score"].append(scores["clip_score"])
                metrics_data["color_diff"].append(scores["color_diff"])

            del original, result, mask_image

        except Exception as e:
            print(f"❌ Error computing metrics for sample {idx}: {e}")
            continue

    # Unload metrics models
    if lpips_fn is not None:
        del lpips_fn
    del clip_model, clip_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute and display final metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    results_summary = {
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Total Samples": NUM_SAMPLES,
        "Painted Samples": len(painted_samples),
        "Not Painted Samples": not_painted_samples,
        "Output Directory": str(inpaint_dir),
    }

    if metrics_data["lpips"]:
        avg_lpips = np.mean(metrics_data["lpips"])
        std_lpips = np.std(metrics_data["lpips"])
        print(f"\n👁️  LPIPS ngoài mask (background) ↓")
        print(f"   Average: {avg_lpips:.4f}  {'✅' if avg_lpips < 0.05 else '⚠️' if avg_lpips < 0.15 else '❌'}  (< 0.05 tốt)")
        print(f"   Std Dev: {std_lpips:.4f}")
        results_summary["LPIPS_background"] = {"avg": round(avg_lpips, 4), "std": round(std_lpips, 4)}

    if metrics_data["boundary"]:
        avg_bc = np.mean(metrics_data["boundary"])
        std_bc = np.std(metrics_data["boundary"])
        print(f"\n🔗 Boundary Consistency ↓")
        print(f"   Average: {avg_bc:.2f}  {'✅' if avg_bc < 8 else '⚠️' if avg_bc < 20 else '❌'}  (< 8 tốt)")
        print(f"   Std Dev: {std_bc:.2f}")
        results_summary["Boundary_Consistency"] = {"avg": round(avg_bc, 2), "std": round(std_bc, 2)}

    if metrics_data["clip_score"]:
        avg_clip = np.mean(metrics_data["clip_score"])
        std_clip = np.std(metrics_data["clip_score"])
        print(f"\n🎯 CLIP Score ↑")
        print(f"   Average: {avg_clip:.2f}  {'✅' if avg_clip > 28 else '⚠️' if avg_clip > 22 else '❌'}  (> 28 tốt)")
        print(f"   Std Dev: {std_clip:.2f}")
        results_summary["CLIP_Score"] = {"avg": round(avg_clip, 2), "std": round(std_clip, 2)}

    if metrics_data["color_diff"]:
        avg_cd = np.mean(metrics_data["color_diff"])
        std_cd = np.std(metrics_data["color_diff"])
        print(f"\n🎨 Color Consistency ↓")
        print(f"   Average: {avg_cd:.2f}  {'✅' if avg_cd < 10 else '⚠️' if avg_cd < 20 else '❌'}  (< 10 tốt)")
        print(f"   Std Dev: {std_cd:.2f}")
        results_summary["Color_Consistency"] = {"avg": round(avg_cd, 2), "std": round(std_cd, 2)}

    print("\n" + "=" * 80)
    print(f"✅ Inpainted images saved to: {inpaint_dir}")
    print("=" * 80)
    
    # Save summary to JSON
    import json
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"📄 Summary saved to: {summary_path}\n")
    
    return results_summary


if __name__ == "__main__":
    main()
