import torch
import gc
from PIL import Image, ImageDraw, ImageFilter
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from IPython.display import display

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights("/home/diffusion/test_diffuser/pytorch_lora_weights1.safetensors")
pipe.set_adapters(["default_0"], adapter_weights=[0.8])

import numpy as np
from PIL import Image, ImageFilter
import cv2
import torch, gc
from diffusers.utils import load_image

# ── UTILS ───────────────────────────────────────────────────────────────────

def resize_max_side(image, max_side=1024):
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image
    new_w = (int(w * scale) // 64) * 64
    new_h = (int(h * scale) // 64) * 64
    return image.resize((new_w, new_h), Image.LANCZOS)


def regularize_mask(mask_image, target_size, roundness_threshold=0.75, sharp=True, feather_radius=15):
    mask_np = np.array(mask_image.convert("L").resize(target_size, Image.NEAREST))
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️  Không tìm thấy contour, dùng mask gốc")
        return mask_image.convert("L").resize(target_size, Image.NEAREST)

    contour     = max(contours, key=cv2.contourArea)
    area        = cv2.contourArea(contour)
    perimeter   = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0

    canvas = np.zeros_like(binary)

    if circularity >= roundness_threshold:
        (cx, cy), (ma, mb), angle = cv2.fitEllipse(contour)
        if min(ma, mb) / max(ma, mb) > 0.90:
            cv2.circle(canvas, (int(cx), int(cy)), int(max(ma, mb) / 2), 255, -1)
            shape = "Circle"
        else:
            cv2.ellipse(canvas, (int(cx), int(cy)),
                        (int(ma / 2), int(mb / 2)), angle, 0, 360, 255, -1)
            shape = "Ellipse"
    else:
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
        cv2.fillPoly(canvas, [box], 255)
        shape = "Rectangle"

    print(f"  Circularity: {circularity:.3f}  →  {shape}")

    if sharp:
        return Image.fromarray(canvas)

    blurred = Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(radius=feather_radius))
    arr = np.array(blurred).astype(float)
    arr = (arr / arr.max() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def inpaint_any_resolution(pipe, init_image, mask_image, prompt, negative_prompt,
                            inpaint_size=1024, crop_padding=128, **pipe_kwargs):
    """
    Crop vùng mask + padding → inpaint ở inpaint_size → paste lại ảnh gốc.
    Hoạt động với mọi resolution, không giới hạn.
    """
    W, H = init_image.size
    mask_np = np.array(mask_image.convert("L"))

    # Bounding box của vùng mask
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        raise ValueError("Mask trống, không có vùng nào được chọn")

    x1 = max(0, xs.min() - crop_padding)
    y1 = max(0, ys.min() - crop_padding)
    x2 = min(W, xs.max() + crop_padding)
    y2 = min(H, ys.max() + crop_padding)

    print(f"  Ảnh gốc    : {W}×{H}")
    print(f"  Vùng crop  : ({x1},{y1}) → ({x2},{y2})  [{x2-x1}×{y2-y1}]")

    # Crop
    crop_img  = init_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_mask = mask_image.convert("L").crop((x1, y1, x2, y2))
    crop_w, crop_h = crop_img.size

    # Resize crop về inpaint_size (chia hết 64)
    scale = inpaint_size / max(crop_w, crop_h)
    inp_w = (int(crop_w * scale) // 64) * 64
    inp_h = (int(crop_h * scale) // 64) * 64

    inp_img  = crop_img.resize((inp_w, inp_h), Image.LANCZOS)
    inp_mask = crop_mask.resize((inp_w, inp_h), Image.NEAREST)

    print(f"  Inpaint tại: {inp_w}×{inp_h}")

    # Inpaint
    result_small = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=inp_img,
        mask_image=inp_mask,
        **pipe_kwargs
    ).images[0]

    # Resize kết quả về kích thước crop gốc
    result_crop = result_small.resize((crop_w, crop_h), Image.LANCZOS)

    # Paste lại đúng vị trí trong ảnh gốc — chỉ vùng mask được thay
    output = init_image.convert("RGB").copy()
    output.paste(result_crop, (x1, y1), mask=crop_mask.resize((crop_w, crop_h), Image.NEAREST))

    return output


# ── CONFIG ───────────────────────────────────────────────────────────────────

img_path  = "G.jpg"
mask_path = "G2_mask.png"

prompt = (
    "Add a modern chair to the scene, placed naturally on the floor. The chair has a clean contemporary design with smooth wooden legs and soft fabric upholstery. It blends naturally with the environment under warm indoor lighting, casting realistic shadows. Ultra realistic, professional interior photography style, sharp focus, 8k resolution."
)
negative_prompt = "blurry, low quality, distorted, deformed, out of frame, bad proportions"

# ── LOAD & PREPROCESS ────────────────────────────────────────────────────────

# Load ảnh gốc KHÔNG resize — giữ full resolution
init_image = load_image(img_path).convert("RGB")
W, H = init_image.size
print(f"Ảnh gốc: {W}×{H}")

print("Regularizing mask...")
mask_image = regularize_mask(
    load_image(mask_path),
    target_size         = (W, H),
    roundness_threshold = 0.75,
    sharp               = True,
    feather_radius      = 15,
)

# ── INPAINTING ───────────────────────────────────────────────────────────────

torch.cuda.empty_cache()
gc.collect()
print("Đang inpainting...")

result = inpaint_any_resolution(
    pipe, init_image, mask_image,
    prompt          = prompt,
    negative_prompt = negative_prompt,
    inpaint_size    = 1024,   # resolution inpaint, tăng lên 1280 nếu VRAM đủ
    crop_padding    = 128,    # padding quanh mask, tăng nếu muốn thêm context
    strength            = 0.99,
    num_inference_steps = 50,
    guidance_scale      = 8.0,
    generator           = torch.Generator("cuda").manual_seed(42)
)

# ── HIỂN THỊ ─────────────────────────────────────────────────────────────────

combined = Image.new("RGB", (W * 3, H))
combined.paste(init_image,              (0,     0))
combined.paste(mask_image.convert("RGB"), (W,   0))
combined.paste(result,                  (W * 2, 0))

print("Bên trái: Ảnh gốc | Giữa: Mask | Phải: Kết quả")
display(combined.resize((1200, int(1200 * H / (W * 3)))))

result.save("result_inpainting_G2.png")
combined.save("comparison_preview_G2.png")
print(f"✅ Đã lưu: 'result_inpainting_G2.png' và 'comparison_preview_G2.png'  ({W}×{H})")