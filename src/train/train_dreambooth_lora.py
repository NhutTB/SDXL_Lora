#!/usr/bin/env python
# coding=utf-8
"""
DreamBooth LoRA fine-tuning for Stable Diffusion XL Inpainting.

Dataset structure expected:
  dataset_dir/
    images/      ← original RGB images   (e.g. 001.jpg / 001.png / 001.webp)
    masks/       ← inpaint masks, grayscale, white = inpaint region
    depths/      ← depth maps (RGB, 3-channel), same stem
    captions/    ← per-image text description, same stem + .txt

  All four folders share the same file *stem*.  The helper _find_file()
  auto-discovers the correct extension.

Usage
-----
# Single GPU
python train_dreambooth_lora_sdxl_inpaint.py --config configs/train_config.yml

# Multi-GPU
accelerate launch --num_processes=4 train_dreambooth_lora_sdxl_inpaint.py \
    --config configs/train_config.yml
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

# ★ SPEED: Import cv2 for fast image loading (3-5x faster than PIL)
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub.utils import insecure_hashlib
from omegaconf import OmegaConf
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageFilter
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_peft_version,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# ★ NEW: scipy for mask dilation (fallback to pure-numpy if unavailable)
try:
    from scipy.ndimage import binary_dilation as _scipy_binary_dilation
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


# ──────────────────────────────────────────────────────────────────────────────
# ★ SPEED: Fast image loading helpers (3-5x faster than PIL)
# ──────────────────────────────────────────────────────────────────────────────

def cv2_imread_rgb(path: str) -> Image.Image | None:
    """
    Fast image loading using cv2 (OpenCV). Converts BGR to RGB and returns PIL Image.
    Falls back to PIL if cv2 fails. ~3-5x faster than PIL.open()
    """
    if not _CV2_AVAILABLE:
        return None
    try:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    except Exception:
        return None


def cv2_imread_grayscale(path: str) -> Image.Image | None:
    """Fast grayscale loading using cv2. Returns PIL Image in 'L' mode."""
    if not _CV2_AVAILABLE:
        return None
    try:
        img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return None
        return Image.fromarray(img_gray, mode="L")
    except Exception:
        return None


def open_image_fast(path, mode: str = "RGB") -> Image.Image:
    """
    Fast image open - tries cv2 first, falls back to PIL.
    mode: "RGB" or "L" (grayscale)
    """
    if mode == "RGB":
        img = cv2_imread_rgb(path)
        if img is not None:
            return img
    elif mode == "L":
        img = cv2_imread_grayscale(path)
        if img is not None:
            return img
    
    # Fallback to PIL
    pil_img = Image.open(path)
    if mode == "RGB":
        return pil_img.convert("RGB")
    elif mode == "L":
        return pil_img.convert("L")
    return pil_img


# ──────────────────────────────────────────────────────────────────────────────
# Memory limit helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_memory_limits(vram_fraction: float = 0.95, ram_reserve_gb: float = 2.0):
    """
    Set VRAM (GPU) hard limit and log system RAM availability for monitoring.
    
    Args:
        vram_fraction: Fraction of total VRAM per GPU (0.0-1.0). Default 0.95 (95%).
        ram_reserve_gb: Reserve this much RAM (GB) from monitoring recommendations. Default 2.0 GB.
    
    Notes:
        - VRAM: Hard limit enforced via CUDA allocator (raises OutOfMemoryError if exceeded)
        - System RAM: Linux kernel 2.6+ does not enforce RLIMIT_RSS.
          We log available/total/recommended for manual monitoring during training.
    """
    # Use standard logging (not accelerate logger, which requires Accelerator init)
    import logging as _logging
    _log = _logging.getLogger(__name__)
    
    # ── VRAM: Hard limit via CUDA allocator ────────────────────────────────
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            torch.cuda.set_per_process_memory_fraction(vram_fraction, device=i)
        _log.info(
            f"VRAM limit set to {vram_fraction*100:.0f}% per GPU ({num_gpus} GPUs) — hard-enforced"
        )
    else:
        _log.warning("CUDA not available — skipping VRAM limit.")
    
    # ── System RAM: Monitoring info (RLIMIT_RSS not enforced on Linux 2.6+) ──
    try:
        import psutil
        mem_info = psutil.virtual_memory()
        total_ram_gb = mem_info.total / (1024**3)
        available_ram_gb = mem_info.available / (1024**3)
    except ImportError:
        # Fallback: read /proc/meminfo if psutil unavailable
        try:
            mem_info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, val = line.split()
                    mem_info[key] = int(val.split()[0])
            total_ram_gb = mem_info.get("MemTotal:", 0) / (1024**2)  # KB to GB
            available_ram_gb = mem_info.get("MemAvailable:", mem_info.get("MemFree:", 0)) / (1024**2)
        except (FileNotFoundError, ValueError, KeyError):
            _log.warning("Could not determine system RAM — skipping RAM info logging.")
            return
    
    if total_ram_gb > 0:
        recommended_limit_gb = total_ram_gb - ram_reserve_gb
        _log.info(
            f"System RAM: {available_ram_gb:.1f} GB available / {total_ram_gb:.1f} GB total. "
            f"Recommended working limit: {recommended_limit_gb:.1f} GB (leaving {ram_reserve_gb:.1f} GB reserve). "
            f"[NOTE: Linux kernel 2.6+ does not enforce RLIMIT_RSS — monitor RAM manually during training]"
        )
    else:
        _log.warning("Total system RAM is 0 — skipping RAM info logging.")


def set_cpu_limit(cpu_fraction: float = 0.90):
    """
    Limit CPU usage by setting process CPU affinity to a subset of available CPUs.
    
    ⚠️  IMPORTANT NOTES:
    
    1. CPU affinity constraints are inherited by child processes in multi-GPU training:
       - With 4 GPUs, accelerate spawns 4 child processes, each inheriting parent's affinity mask
       - Result: All 4 processes compete for the SAME CPU subset → not efficient
       - Workaround: This is OK for most LoRA training (CPU is NOT the bottleneck anyway)
    
    2. CPU affinity does NOT hard-limit CPU usage:
       - It only restricts which CPUs a process CAN use, not total CPU %
       - It's a scheduling constraint, not a hard resource limit (like VRAM)
       - Multiple processes can still saturate those CPUs
    
    3. Actual root cause of server overload (from previous session):
       - NOT CPU exhaustion — was RAM OOM from bulk dataset preload ✓ (fixed via lazy loading)
       - CPU control from accelerate (OMP_NUM_THREADS=1) + num_workers: 2-4 is sufficient
    
    4. This function as a protective layer:
       - Adds extra safeguard against CPU thrashing
       - No harm if GPU training is I/O bottlenecked (usual case with LoRA)
       - Helps prevent system responsiveness degradation in edge cases

    Args:
        cpu_fraction: Fraction of total CPUs to use (0.0-1.0). Default 0.90 (90%).
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)
    
    try:
        import psutil
        
        total_cpus = os.cpu_count() or psutil.cpu_count()
        num_cpus_to_use = max(1, int(total_cpus * cpu_fraction))
        
        # Set CPU affinity to use only the first num_cpus_to_use CPUs
        available_cpus = list(range(num_cpus_to_use))
        psutil.Process().cpu_affinity(available_cpus)
        
        _log.info(
            f"CPU limit set to {cpu_fraction*100:.0f}% ({num_cpus_to_use}/{total_cpus} CPUs)"
        )
    except ImportError:
        _log.warning("psutil not available — skipping CPU limit.")
    except Exception as e:
        _log.warning(f"Failed to set CPU affinity ({e}). Continuing without CPU limit.")


# ──────────────────────────────────────────────────────────────────────────────
# DeepSpeed ZeRO-3 helper
# ──────────────────────────────────────────────────────────────────────────────

def _zero3_gather(model, accelerator, modifier_rank=0):
    """Context manager for gathering ZeRO-3 partitioned parameters.
    All ranks must enter/exit together (collective op). modifier_rank=0 means
    only rank-0 gets write access; others participate but read zeros."""
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        try:
            import deepspeed
            params = list(model.parameters())
            if params:
                return deepspeed.zero.GatheredParameters(params, modifier_rank=modifier_rank)
        except ImportError:
            pass
    return nullcontext()


# ──────────────────────────────────────────────────────────────────────────────
# ★ NEW: RAM monitoring daemon (prevent OOM killer from killing other processes)
# ──────────────────────────────────────────────────────────────────────────────

_RAM_MONITOR_STOP_EVENT: threading.Event | None = None
_RAM_MONITOR_THREAD: threading.Thread | None = None

def start_ram_monitor_daemon(threshold_pct: float = 85.0, high_count_limit: int = 3, check_interval: float = 1.0):
    """
    Start a daemon thread to monitor system RAM usage.
    
    If RAM usage exceeds threshold_pct for high_count_limit consecutive checks,
    kill the process group with SIGTERM to prevent OOM killer from randomly killing processes.
    
    Args:
        threshold_pct: RAM usage threshold (default 85%)
        high_count_limit: Number of consecutive checks before shutdown (default 3)
        check_interval: Seconds between checks (default 5s)
    """
    import os
    import signal
    import logging as _logging        
    _log = _logging.getLogger(__name__) 
    global _RAM_MONITOR_STOP_EVENT, _RAM_MONITOR_THREAD
    
    # Create event and thread-local state
    stop_event = threading.Event()
    
    def _monitor_loop():
        high_count = 0
        import psutil
        
        while not stop_event.is_set():
            try:
                mem_pct = psutil.virtual_memory().percent
                if mem_pct > threshold_pct:
                    high_count += 1
                    if high_count >= high_count_limit:
                        _log.warning(
                            f"[RAM Monitor] RAM usage {mem_pct:.1f}% exceeded {threshold_pct}% "
                            f"for {high_count_limit} consecutive checks. Killing process group..."
                        )
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                else:
                    high_count = 0  # Reset counter if back below threshold
            except ImportError:
                _log.warning("[RAM Monitor] psutil not available, skipping RAM monitoring")
                return
            except Exception as e:
                _log.warning(f"[RAM Monitor] Monitoring error: {e}")
            
            stop_event.wait(timeout=check_interval)
    
    # Save to globals for cleanup access if needed
    _RAM_MONITOR_STOP_EVENT = stop_event
    
    # Start daemon thread (will be killed when main thread exits)
    _RAM_MONITOR_THREAD = threading.Thread(
        target=_monitor_loop,
        daemon=True,
        name="RAMMonitorDaemon"
    )
    _RAM_MONITOR_THREAD.start()
    _log.info("[RAM Monitor] Started daemon thread (threshold={:.0f}%, checks=every {:.1f}s)".format(threshold_pct, check_interval))


# ──────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str):
    return OmegaConf.load(path)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str | None,
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    """Encode prompt with both SDXL text encoders."""
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            text_input_ids = tokenize_prompt(tokenizers[i], prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds_list.append(prompt_embeds.view(bs_embed, seq_len, -1))

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# ──────────────────────────────────────────────────────────────────────────────
# Train / Val / Test split  (80 / 10 / 10)
# ──────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_file(directory: Path, stem: str, exts: list[str]) -> Path | None:
    """Search for *stem* + any of *exts* inside *directory*."""
    for ext in exts:
        candidate = directory / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def make_splits(
    images_dir: str,
    masks_dir: str,
    depths_dir: str,
    captions_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float   = 0.1,
    seed: int          = 42,
) -> dict[str, list[Path]]:
    """
    Scan *images_dir* and keep images. Masks, depths, and captions are optional
    (Dataset class handles missing files with fallbacks).
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    images_root   = Path(images_dir)
    masks_root    = Path(masks_dir)
    depths_root   = Path(depths_dir)
    captions_root = Path(captions_dir)

    _img_ext_list  = list(_IMG_EXTS)
    _depth_exts    = _img_ext_list
    _caption_exts  = [".txt"]

    valid: list[Path] = []
    skipped: dict[str, int] = {"mask": 0, "depth": 0, "caption": 0}

    for p in sorted(images_root.iterdir()):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        stem = p.stem

        if _find_file(masks_root,    stem, _img_ext_list) is None:
            skipped["mask"] += 1
            continue
        if _find_file(depths_root,   stem, _depth_exts)   is None:
            skipped["depth"] += 1
            continue
        if _find_file(captions_root, stem, _caption_exts)  is None:
            skipped["caption"] += 1
            continue

        valid.append(p)

    if not valid:
        raise ValueError(
            f"No valid (image+mask+depth+caption) quads found in {images_dir}. "
            f"Skipped: mask={skipped['mask']}, depth={skipped['depth']}, "
            f"caption={skipped['caption']}"
        )

    for reason, count in skipped.items():
        if count:
            logger.warning(f"make_splits: {count} images skipped (missing {reason})")

    rng = random.Random(seed)
    shuffled = valid.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train: n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    logger.info(
        f"Dataset split (seed={seed}): "
        f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}  "
        f"total={n}"
    )
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# LoRA / DoRA config
# ──────────────────────────────────────────────────────────────────────────────

def get_lora_config(rank: int, alpha: int, dropout: float, use_dora: bool, target_modules: list):
    base = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if use_dora:
        if is_peft_version("<", "0.9.0"):
            raise ValueError("DoRA requires peft >= 0.9.0. Run: pip install -U peft")
        base["use_dora"] = True
    return LoraConfig(**base)


# ──────────────────────────────────────────────────────────────────────────────
# Model card
# ──────────────────────────────────────────────────────────────────────────────

def save_model_card(output_dir: str, base_model: str, dataset_dir: str,
                    lora_enabled: bool, use_dora: bool,
                    instance_prompt: str | None = None,
                    validation_prompt: str | None = None) -> None:
    adapter_type = "DoRA" if use_dora else ("LoRA" if lora_enabled else "Full fine-tune")
    card = f"""---
base_model: {base_model}
tags:
  - stable-diffusion-xl
  - inpainting
  - interior-design
  - dreambooth
  - {adapter_type.lower()}
license: openrail++
---

# SDXL Inpainting DreamBooth {adapter_type} – Interior Design

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) using
DreamBooth with {adapter_type} adapters on an interior-design inpainting dataset.

Training dataset: `{dataset_dir}`

{"**Trigger word:** `" + instance_prompt + "`" if instance_prompt else ""}
{"**Validation prompt:** `" + validation_prompt + "`" if validation_prompt else ""}
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)


# ──────────────────────────────────────────────────────────────────────────────
# ★ NEW: Mask dilation helper
# ──────────────────────────────────────────────────────────────────────────────

def dilate_mask_pct(
    mask_arr: np.ndarray,
    dilation_pct: float = 0.10,
    max_area_pct: float = 0.70,
) -> np.ndarray:
    """
    Dilate a binary mask (H×W, dtype uint8, values 0 or 255) by a kernel whose
    radius is dilation_pct * min(H, W) pixels.  The result is then capped so
    the dilated region never exceeds max_area_pct of the total image area.

    Parameters
    ----------
    mask_arr     : H×W uint8 array (0 = background, 255 = masked)
    dilation_pct : fraction of min(H,W) used as dilation radius  (default 0.10)
    max_area_pct : maximum fraction of image that may be masked   (default 0.70)

    Returns
    -------
    H×W uint8 array (0 or 255)
    """
    h, w  = mask_arr.shape
    binary = mask_arr > 127

    # Skip if mask is blank
    if not binary.any():
        return mask_arr

    # Radius in pixels
    radius = max(2, int(min(h, w) * dilation_pct))

    if _SCIPY_AVAILABLE:
        # Circular (elliptical) structuring element
        y, x   = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        struct = (x ** 2 + y ** 2) <= radius ** 2
        dilated = _scipy_binary_dilation(binary, structure=struct)
    else:
        # Pure-numpy square dilation (fallback)
        from numpy.lib.stride_tricks import sliding_window_view
        pad     = radius
        padded  = np.pad(binary.astype(np.uint8), pad, mode="constant")
        windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
        dilated = windows.max(axis=(-2, -1)).astype(bool)

    # ── Cap total masked area at max_area_pct ─────────────────────────────
    max_pixels = int(h * w * max_area_pct)
    if dilated.sum() > max_pixels:
        # Keep the original mask if dilation would push us over the cap.
        # (We never shrink a mask that was already over the cap.)
        if binary.sum() <= max_pixels:
            dilated = binary
        else:
            dilated = binary  # already over cap — leave as-is

    return (dilated * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# ★ NEW: Auxiliary loss computer
# ──────────────────────────────────────────────────────────────────────────────

class AuxiliaryLossComputer:
    """
    Lazily loads frozen helper models (VGG16, CLIP ViT-B/32, MiDaS) on first
    use.  Computes pixel, perceptual, CLIP, boundary, depth, and semantic losses
    in pixel space by first reconstructing the predicted clean image from the
    UNet output.

    All sub-models are kept frozen (eval mode, no grad through their weights)
    but gradients DO flow through the decoded predictions back to the LoRA
    parameters.

    Install optional deps:
        pip install torchvision transformers torch    (VGG + CLIP)
        # MiDaS is fetched via torch.hub on first use
    """

    # VGG ImageNet normalisation constants
    _VGG_MEAN = [0.485, 0.456, 0.406]
    _VGG_STD  = [0.229, 0.224, 0.225]

    # CLIP ImageNet normalisation constants
    _CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
    _CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, device: torch.device, weight_dtype: torch.dtype):
        self.device       = device
        self.weight_dtype = weight_dtype

        # Lazy-loaded models (None = not yet loaded, "failed" = load error)
        self._vgg16           = None
        self._clip_vision     = None
        self._depth_model     = None
        self._depth_transform = None

    # ── Lazy model loaders ────────────────────────────────────────────────

    @property
    def vgg16(self):
        """VGG16 feature extractor (up to relu4_3 = layer index 23)."""
        if self._vgg16 is None:
            try:
                from torchvision.models import vgg16, VGG16_Weights
                net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]
                net = net.to(self.device).eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                self._vgg16 = net
                logger.info("AuxLoss: VGG16 perceptual model loaded (relu4_3).")
            except Exception as e:
                logger.warning(f"AuxLoss: VGG16 failed to load ({e}). "
                               "Perceptual + semantic losses disabled.")
                self._vgg16 = "failed"
        return self._vgg16

    @property
    def clip_vision(self):
        """CLIP ViT-B/32 vision encoder (shared for CLIP + semantic losses)."""
        if self._clip_vision is None:
            try:
                from transformers import CLIPVisionModel
                m = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", low_cpu_mem_usage=False)
                m = m.to(self.device).eval()
                for p in m.parameters():
                    p.requires_grad_(False)
                self._clip_vision = m
                logger.info("AuxLoss: CLIP ViT-B/32 vision model loaded.")
            except Exception as e:
                logger.warning(f"AuxLoss: CLIP model failed ({e}). "
                               "CLIP + semantic losses disabled.")
                self._clip_vision = "failed"
        return self._clip_vision

    @property
    def depth_model(self):
        """MiDaS small depth estimator (loaded via torch.hub)."""
        if self._depth_model is None:
            try:
                model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                xforms = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                )
                model = model.to(self.device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                self._depth_model     = model
                self._depth_transform = xforms.small_transform
                logger.info("AuxLoss: MiDaS-small depth model loaded.")
            except Exception as e:
                logger.warning(f"AuxLoss: MiDaS failed ({e}). Depth loss disabled.")
                self._depth_model     = "failed"
                self._depth_transform = None
        return self._depth_model, self._depth_transform

    # ── Tensor helpers ────────────────────────────────────────────────────

    @staticmethod
    def _denorm(x: torch.Tensor) -> torch.Tensor:
        """Diffusion-space [-1, 1]  →  [0, 1]."""
        return (x * 0.5 + 0.5).clamp(0, 1)

    def _norm_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] RGB  →  VGG ImageNet normalised."""
        mean = torch.tensor(self._VGG_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self._VGG_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def _norm_clip(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] RGB  →  CLIP normalised."""
        mean = torch.tensor(self._CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self._CLIP_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    # ── Individual loss functions ─────────────────────────────────────────

    def _pixel_loss(
        self,
        pred: torch.Tensor,   # [B, 3, H, W] in [0, 1]
        gt:   torch.Tensor,
        mask: torch.Tensor,   # [B, 1, H, W] in {0, 1}
    ) -> torch.Tensor:
        """
        L1 loss restricted to the masked (inpainted) region.
        Falls back to global L1 if mask is empty.
        """
        diff = (pred - gt).abs()
        denom = mask.sum() * diff.shape[1] + 1e-8
        if mask.sum() > 0:
            return (diff * mask).sum() / denom
        return diff.mean()

    def _perceptual_loss(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> torch.Tensor:
        """VGG16 relu4_3 feature MSE (full image — context matters for style)."""
        vgg = self.vgg16
        if vgg == "failed":
            return torch.tensor(0.0, device=pred.device)
        feat_pred = vgg(self._norm_vgg(pred))
        with torch.no_grad():
            feat_gt = vgg(self._norm_vgg(gt))
        return F.mse_loss(feat_pred.float(), feat_gt.float())

    def _clip_loss(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> torch.Tensor:
        """
        1 - cosine similarity between CLIP [CLS] embeddings.
        Penalises semantic drift of the inpainted region.
        """
        model = self.clip_vision
        if model == "failed":
            return torch.tensor(0.0, device=pred.device)
        # CLIP expects 224×224 images
        p = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        g = F.interpolate(gt,   size=(224, 224), mode="bilinear", align_corners=False)
        p_feat = model(pixel_values=self._norm_clip(p)).last_hidden_state[:, 0]  # [CLS]
        with torch.no_grad():
            g_feat = model(pixel_values=self._norm_clip(g)).last_hidden_state[:, 0]
        cos = F.cosine_similarity(p_feat.float(), g_feat.float(), dim=-1)
        return (1.0 - cos).mean()

    def _boundary_loss(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Laplacian edge-coherence loss on the narrow band around the mask
        boundary.  Encourages seamless blending where the inpainted region
        meets the preserved content.
        """
        # Laplacian kernel applied channel-by-channel
        lap_k = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
            dtype=pred.dtype, device=pred.device,
        ).view(1, 1, 3, 3).expand(3, 1, 3, 3)

        # Boundary = dilation(mask) - mask  (thin ring, 5-px wide)
        boundary = (
            F.max_pool2d(mask, kernel_size=5, stride=1, padding=2) - mask
        ).clamp(0, 1)

        edges_pred = F.conv2d(pred, lap_k, padding=1, groups=3)
        with torch.no_grad():
            edges_gt = F.conv2d(gt, lap_k, padding=1, groups=3)

        diff  = (edges_pred - edges_gt).abs()
        denom = boundary.sum() * diff.shape[1] + 1e-8
        if boundary.sum() > 0:
            return (diff * boundary).sum() / denom
        return diff.mean()

    def _depth_loss(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MiDaS-small depth-consistency loss on the masked region.
        Ensures the inpainted region matches the depth profile of the scene.
        """
        model, _ = self.depth_model
        if model == "failed" or model is None:
            return torch.tensor(0.0, device=pred.device)

        # Resize to 256 for speed
        size = (256, 256)
        p = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        g = F.interpolate(gt,   size=size, mode="bilinear", align_corners=False)
        m = F.interpolate(mask, size=size, mode="nearest")

        # MiDaS trained on BGR input
        with torch.no_grad():
            d_gt   = model(g.flip(1)).unsqueeze(1)   # [B, 1, H, W]
        d_pred = model(p.flip(1)).unsqueeze(1)

        if m.sum() > 0:
            return F.l1_loss(d_pred.float() * m, d_gt.float() * m) / (m.mean() + 1e-8)
        return F.l1_loss(d_pred.float(), d_gt.float())

    def _semantic_loss(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Patch-level CLIP token MSE — captures high-level semantic structure
        beyond the [CLS] summary vector used in _clip_loss.
        Reuses the same (already-loaded) CLIP model.
        """
        model = self.clip_vision
        if model == "failed":
            return torch.tensor(0.0, device=pred.device)
        p = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        g = F.interpolate(gt,   size=(224, 224), mode="bilinear", align_corners=False)
        p_tokens = model(pixel_values=self._norm_clip(p)).last_hidden_state   # [B, 197, 768]
        with torch.no_grad():
            g_tokens = model(pixel_values=self._norm_clip(g)).last_hidden_state
        return F.mse_loss(p_tokens.float(), g_tokens.float())

    # ── Public interface ──────────────────────────────────────────────────

    def reconstruct_pred_x0(
        self,
        model_pred:    torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps:     torch.Tensor,
        noise_scheduler,
        do_edm:        bool,
    ) -> torch.Tensor:
        """
        Reconstruct the predicted clean latent x₀ from the UNet output.

        For EDM: model_pred has already been post-conditioned into x₀ space
                 by the calling code (epsilon → x₀ transform applied there).
        For DDPM epsilon-prediction: x₀ = (xₜ − √(1−ᾱₜ)·ε) / √ᾱₜ
        For DDPM v-prediction:       x₀ = √ᾱₜ·xₜ − √(1−ᾱₜ)·v
        """
        if do_edm:
            # Already in x₀ space after EDM post-conditioning
            return model_pred.float()

        alphas_cp = noise_scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=torch.float32
        )
        a = alphas_cp[timesteps].view(-1, 1, 1, 1)
        b = (1.0 - a)

        pred_type = noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            pred_x0 = (noisy_latents.float() - b.sqrt() * model_pred.float()) / a.sqrt()
        elif pred_type == "v_prediction":
            pred_x0 = a.sqrt() * noisy_latents.float() - b.sqrt() * model_pred.float()
        else:
            raise ValueError(f"Unknown prediction_type: {pred_type}")

        return pred_x0.clamp(-5.0, 5.0)   # guard against numerical blow-up

    def compute(
        self,
        model_pred:    torch.Tensor,   # UNet output (instance half only)
        noisy_latents: torch.Tensor,   # x_t            (instance half only)
        timesteps:     torch.Tensor,   # t              (instance half only)
        gt_latents:    torch.Tensor,   # clean latents  (instance half only)
        mask_latent:   torch.Tensor,   # downsampled mask [B, 1, h, w]
        vae:           AutoencoderKL,
        weights,                       # OmegaConf loss_weights node
        do_edm:        bool,
        noise_scheduler,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute all enabled auxiliary losses and return
        (total_weighted_aux_loss, {metric_name: scalar}).

        Losses that are disabled (weight == 0) are skipped entirely.
        VAE decode is only performed when at least one pixel-space loss is active.
        """
        total = torch.tensor(0.0, device=model_pred.device)
        log:  dict[str, float] = {}

        need_decode = any([
            weights.pixel_weight      > 0,
            weights.perceptual_weight > 0,
            weights.clip_weight       > 0,
            weights.boundary_weight   > 0,
            weights.depth_weight      > 0,
            weights.semantic_weight   > 0,
        ])

        if not need_decode:
            return total, log

        # ── Reconstruct predicted clean latent ───────────────────────────
        pred_x0 = self.reconstruct_pred_x0(
            model_pred, noisy_latents, timesteps, noise_scheduler, do_edm
        )

        # ── Decode to pixel space ─────────────────────────────────────────
        # VAE is frozen → no weight updates, but gradients pass through
        # pred decode (training supervision), not through gt decode.
        # ★ Downscale latents trước khi decode → tiết kiệm VRAM
        scale = getattr(weights, 'aux_resolution_scale', 1.0)
        if scale < 1.0:
            pred_x0    = F.interpolate(pred_x0,    scale_factor=scale, mode="bilinear", align_corners=False)
            gt_latents = F.interpolate(gt_latents, scale_factor=scale, mode="bilinear", align_corners=False)
            mask_latent = F.interpolate(mask_latent, scale_factor=scale, mode="nearest")

        # ★ CRITICAL: Clear cache before VAE decode to avoid OOM by maximizing contiguous headroom
        torch.cuda.empty_cache()

        inv_sf = 1.0 / vae.config.scaling_factor
        
        # ★ NEW: Decode samples one-by-one to minimize peak VRAM during VAE decode
        # (Decoding 1024px samples at once is the main cause of OOM on 32GB)
        p_list = []
        for i in range(pred_x0.shape[0]):
            lat = pred_x0[i:i+1].to(vae.dtype) * inv_sf
            p_chunk = vae.decode(lat).sample
            p_list.append(self._denorm(p_chunk))
        pred_px = torch.cat(p_list, dim=0).clamp(0, 1).float()

        with torch.no_grad():
            g_list = []
            for i in range(gt_latents.shape[0]):
                lat = gt_latents[i:i+1].to(vae.dtype) * inv_sf
                g_chunk = vae.decode(lat).sample
                g_list.append(self._denorm(g_chunk))
            gt_px = torch.cat(g_list, dim=0).clamp(0, 1).float()
        # Upsample latent-space mask → pixel space
        mask_px = F.interpolate(
            mask_latent.float(), size=pred_px.shape[-2:], mode="nearest"
        )

        # ── Pixel L1 ──────────────────────────────────────────────────────
        if weights.pixel_weight > 0:
            l = self._pixel_loss(pred_px, gt_px, mask_px)
            total = total + weights.pixel_weight * l
            log["train/loss_pixel"] = l.detach().item()

        # ── Perceptual (VGG) ──────────────────────────────────────────────
        if weights.perceptual_weight > 0:
            l = self._perceptual_loss(pred_px, gt_px)
            total = total + weights.perceptual_weight * l
            log["train/loss_perceptual"] = l.detach().item()

        # ── CLIP semantic similarity ───────────────────────────────────────
        if weights.clip_weight > 0:
            l = self._clip_loss(pred_px, gt_px)
            total = total + weights.clip_weight * l
            log["train/loss_clip"] = l.detach().item()

        # ── Boundary edge coherence ───────────────────────────────────────
        if weights.boundary_weight > 0:
            l = self._boundary_loss(pred_px, gt_px, mask_px)
            total = total + weights.boundary_weight * l
            log["train/loss_boundary"] = l.detach().item()

        # ── Depth consistency (MiDaS) ─────────────────────────────────────
        if weights.depth_weight > 0:
            l = self._depth_loss(pred_px, gt_px, mask_px)
            total = total + weights.depth_weight * l
            log["train/loss_depth"] = l.detach().item()

        # ── Semantic patch features (CLIP tokens) ─────────────────────────
        if weights.semantic_weight > 0:
            l = self._semantic_loss(pred_px, gt_px)
            total = total + weights.semantic_weight * l
            log["train/loss_semantic"] = l.detach().item()

        return total, log


# ──────────────────────────────────────────────────────────────────────────────
# DreamBooth Inpainting Dataset  (4-folder structure)
# ──────────────────────────────────────────────────────────────────────────────

class DreamBoothInpaintingDataset(Dataset):
    """
    Loads (image, mask, depth, caption) quads from 4 parallel directories.

    __getitem__ returns a dict with 6 keys:
      pixel_values   : [3, H, W]  float32, range [-1, 1]  (for VAE)
      masks          : [1, H, W]  float32, range [0, 1]   (1 = inpaint)
      masked_images  : [3, H, W]  float32, range [-1, 1]  (image * (mask < 0.5))
      depth_values   : [3, H, W]  float32, range [0, 1]   (depth map, no extra norm)
      input_ids_one  : [seq_len]  long  (CLIP tokenizer 1)
      input_ids_two  : [seq_len]  long  (CLIP tokenizer 2)
    """

    _IMG_EXTS_LIST    = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    _CAPTION_EXTS     = [".txt"]

    def __init__(
        self,
        images_dir:    str,
        masks_dir:     str,
        depths_dir:    str,
        captions_dir:  str,
        tokenizer_one,
        tokenizer_two,
        file_list:     list[Path],
        fallback_caption: str = "",
        class_prompt:  str | None = None,
        class_data_dir: str | None = None,
        class_num:     int | None = None,
        size:          int = 1024,
        repeats:       int = 1,
        center_crop:   bool = False,
        random_flip:   bool = False,
        mask_min_area: float = 0.1,
        mask_max_area: float = 0.5,
        mask_dilation_pct: float = 0.10,
        mask_max_area_pct: float = 0.70,
        preload_to_ram: bool = False,  # ★ NEW: force preload all data to RAM
    ):
        self.size              = size
        self.center_crop       = center_crop
        self.random_flip       = random_flip
        self.fallback_caption  = fallback_caption
        self.class_prompt      = class_prompt
        self.mask_min_area     = mask_min_area
        self.mask_max_area     = mask_max_area
        self.mask_dilation_pct = mask_dilation_pct
        self.mask_max_area_pct = mask_max_area_pct
        self.tokenizer_one     = tokenizer_one
        self.tokenizer_two     = tokenizer_two

        self.masks_root    = Path(masks_dir)
        self.depths_root   = Path(depths_dir)
        self.captions_root = Path(captions_dir)

        for label, path in [
            ("masks_dir",    self.masks_root),
            ("depths_dir",   self.depths_root),
            ("captions_dir", self.captions_root),
        ]:
            if not path.exists():
                raise ValueError(f"{label} does not exist: {path}")
        if not file_list:
            raise ValueError("file_list is empty — nothing to load!")

        self.instance_paths: list[Path] = list(
            itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in file_list)
        )

        # ── Image transforms ────────────────────────────────────────────────
        self.train_resize  = transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS)
        self.train_crop    = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip_tf = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor     = transforms.ToTensor()   # → [0, 1]

        # ── Scan image headers (lazy — no pixel data) ────────────────────────
        self.original_sizes: list[tuple[int, int]] = []
        logger.info(f"Scanning {len(self.instance_paths)} images (lazy loading) …")
        for img_path in tqdm(
            self.instance_paths,
            desc="Scanning image sizes",
            unit="img",
            dynamic_ncols=True,
            disable=not getattr(__import__('os'), 'isatty', lambda x: True)(0),
        ):
            try:
                # ★ SPEED: Use cv2 for faster header reading
                if _CV2_AVAILABLE:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        self.original_sizes.append((h, w))
                    else:
                        raise ValueError("cv2.imread returned None")
                else:
                    # Fallback to PIL
                    with Image.open(img_path) as pil_img:
                        self.original_sizes.append((pil_img.height, pil_img.width))
            except Exception as e:
                logger.warning(f"Failed to read image header {img_path}: {e}. Using (1024, 1024).")
                self.original_sizes.append((size, size))

        self.num_instance_images = len(self.instance_paths)
        self._length = self.num_instance_images

        # ── Optional prior-preservation class images ─────────────────────────
        self.class_data_root = None
        if class_data_dir is not None:
            self.class_data_root = Path(class_data_dir)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            class_paths = sorted(
                p for p in self.class_data_root.iterdir()
                if p.suffix.lower() in _IMG_EXTS
            )
            self.num_class_images = min(len(class_paths), class_num) if class_num else len(class_paths)
            self.class_paths = class_paths[: self.num_class_images]
            self._length = max(self.num_class_images, self.num_instance_images)

        self.class_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # ★ NEW: In-memory cache to reduce CPU bottleneck ──────────────────────
        # Decision: based entirely on config, no auto-threshold
        self._image_cache: dict = {}  # stem -> PIL Image
        self._mask_cache:  dict = {}  # stem -> torch.Tensor [1, H, W]
        self._depth_cache: dict = {}  # stem -> torch.Tensor [3, H, W]
        self._tokens_cache: dict = {}  # caption -> (ids_one, ids_two)
        self._latent_cache: dict = {}  # stem -> torch.Tensor [4, 128, 128] (bfloat16)
        
        # ★ FIXED: Respect config exclusively (no threshold fallback)
        self._preload_all = preload_to_ram
        
        if self._preload_all:
            logger.info(f"★ Preloading {self.num_instance_images} images to RAM (cache mode) …")
            self._preload_all_data()

    def _preload_all_data(self):
        """Preload all images/masks/depths into RAM for zero I/O overhead."""
        for img_path in tqdm(
            self.instance_paths,
            desc="Preloading to RAM",
            unit="img",
            dynamic_ncols=True,
            disable=not getattr(__import__('os'), 'isatty', lambda x: True)(0),
        ):
            stem = img_path.stem
            if stem in self._image_cache:
                continue  # already loaded (from repeats)
            
            # Load & cache image
            try:
                # ★ SPEED: Use fast loader
                img_orig = open_image_fast(str(img_path), mode="RGB") or Image.open(img_path).convert("RGB")
                img_resized = self.train_resize(img_orig)
                self._image_cache[stem] = img_resized
            except Exception as e:
                logger.warning(f"Failed to preload image {img_path}: {e}")
            
            # Load & cache mask
            mask_path = self._find_file(self.masks_root, stem, self._IMG_EXTS_LIST)
            try:
                if mask_path:
                    self._mask_cache[stem] = self._load_mask(mask_path)
                else:
                    self._mask_cache[stem] = None
            except Exception as e:
                logger.warning(f"Failed to preload mask {stem}: {e}")
            
            # Load & cache depth
            depth_path = self._find_file(self.depths_root, stem, self._IMG_EXTS_LIST)
            try:
                if depth_path:
                    # ★ SPEED: Use fast loader
                    d = open_image_fast(str(depth_path), mode="RGB") or Image.open(depth_path).convert("RGB")
                    d = d.resize((self.size, self.size), Image.LANCZOS)
                    self._depth_cache[stem] = self.to_tensor(d)
                else:
                    self._depth_cache[stem] = None
            except Exception as e:
                logger.warning(f"Failed to preload depth {stem}: {e}")

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _find_file(directory: Path, stem: str, exts: list[str]) -> Path | None:
        """Return the first existing file matching *stem* + any of *exts*."""
        for ext in exts:
            candidate = directory / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def _tokenize(self, text: str):
        """Tokenize *text* for both CLIP encoders, return (ids_one, ids_two)."""
        def _tok(tokenizer, txt):
            return tokenizer(
                txt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]   # shape: [seq_len]
        return _tok(self.tokenizer_one, text), _tok(self.tokenizer_two, text)

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """Load grayscale mask → [1, H, W] float32 in [0, 1]. White = inpaint."""
        size = self.size
        # ★ SPEED: Use fast image loader for mask
        m = open_image_fast(str(mask_path), mode="L") or Image.open(mask_path).convert("L")
        m = m.resize((size, size), Image.NEAREST)
        m_arr = np.array(m, dtype=np.uint8)
        if not (m_arr > 127).any():
            return self._random_box_mask(size)
        m_arr = dilate_mask_pct(
            m_arr,
            dilation_pct=self.mask_dilation_pct,
            max_area_pct=self.mask_max_area_pct,
        )
        return torch.from_numpy((m_arr > 127).astype(np.float32)).unsqueeze(0)

    def _random_box_mask(self, size: int) -> torch.Tensor:
        area      = size * size
        mask_area = random.uniform(self.mask_min_area, self.mask_max_area) * area
        h  = int(random.uniform(0.2, 0.8) * size)
        w  = min(int(mask_area / max(h, 1)), size)
        y0 = random.randint(0, size - h)
        x0 = random.randint(0, size - w)
        mask = np.zeros((size, size), dtype=np.float32)
        mask[y0: y0 + h, x0: x0 + w] = 1.0
        return torch.from_numpy(mask).unsqueeze(0)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        i        = index % self.num_instance_images
        img_path = self.instance_paths[i]
        stem     = img_path.stem

        # ── 1. Caption → tokenize for both CLIP encoders ─────────────────────
        cap_path = self._find_file(self.captions_root, stem, self._CAPTION_EXTS)
        if cap_path is not None:
            try:
                caption = cap_path.read_text(encoding="utf-8").strip()
            except Exception:
                caption = self.fallback_caption
        else:
            caption = self.fallback_caption

        # ★ NEW: Cache tokenization results
        if caption not in self._tokens_cache:
            self._tokens_cache[caption] = self._tokenize(caption)
        input_ids_one, input_ids_two = self._tokens_cache[caption]

        # ── 2. Image (RGB) → resize → optional flip+crop → [-1, 1] ──────────
        # ★ NEW: Load from cache if preloaded
        if self._preload_all and stem in self._image_cache:
            image = self._image_cache[stem].copy()  # copy to allow independent flipping
        else:
            # ★ SPEED: Use fast image loader (cv2) instead of PIL - 3-5x faster
            image = open_image_fast(img_path, mode="RGB") or Image.open(img_path).convert("RGB")
            image = self.train_resize(image)

        do_flip = self.random_flip and random.random() < 0.5
        if do_flip:
            image = self.train_flip_tf(image)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width  - self.size) / 2.0)))
        else:
            y1, x1, _, _ = self.train_crop.get_params(image, (self.size, self.size))

        image = crop(image, y1, x1, self.size, self.size)
        pv    = self.to_tensor(image) * 2.0 - 1.0   # [0,1] → [-1, 1]

        # ── 3. Mask (L) → resize NEAREST → [0, 1], 1-channel ────────────────
        # ★ NEW: Load from cache if preloaded
        if self._preload_all and stem in self._mask_cache:
            mask = self._mask_cache[stem]
            if mask is None:
                mask = self._random_box_mask(self.size)
        else:
            mask_path = self._find_file(self.masks_root, stem, self._IMG_EXTS_LIST)
            mask      = self._load_mask(mask_path) if mask_path else self._random_box_mask(self.size)

        # ── 4. Depth (RGB) → resize → [0, 1], 3-channel ─────────────────────
        # ★ NEW: Load from cache if preloaded
        if self._preload_all and stem in self._depth_cache:
            depth = self._depth_cache[stem]
            if depth is None:
                depth = torch.zeros(3, self.size, self.size)
            if do_flip:
                depth = torch.flip(depth, dims=[2])  # flip along width
        else:
            depth_path = self._find_file(self.depths_root, stem, self._IMG_EXTS_LIST)
            if depth_path is not None:
                # ★ SPEED: Use fast image loader for depth
                depth_img = open_image_fast(str(depth_path), mode="RGB") or Image.open(depth_path).convert("RGB")
                depth_img = depth_img.resize((self.size, self.size), Image.LANCZOS)
                if do_flip:
                    from PIL import ImageOps
                    depth_img = ImageOps.mirror(depth_img)
                depth = self.to_tensor(depth_img)   # [0, 1]
            else:
                depth = torch.zeros(3, self.size, self.size)

        # ── 5. Masked image ───────────────────────────────────────────────────
        masked_image = pv * (mask < 0.5)   # black-out the inpaint region

        example = {
            "pixel_values":  pv,
            "masks":         mask,
            "masked_images": masked_image,
            "depth_values":  depth,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            # metadata for SDXL micro-conditioning
            "original_size":  self.original_sizes[i],
            "crop_top_left":  (y1, x1),
            # ★ NEW: Latent cache (if available)
            "latent_cached":  stem in self._latent_cache,
            "stem": stem,  # Store stem for latent lookup during training
        }

        # ── Optional prior-preservation class image ───────────────────────────
        if self.class_data_root:
            j       = index % self.num_class_images
            # ★ SPEED: Use fast image loader for class images
            cls_img = open_image_fast(str(self.class_paths[j]), mode="RGB") or Image.open(self.class_paths[j]).convert("RGB")
            cls_pv  = self.class_transform(cls_img)
            cls_mask = self._random_box_mask(self.size)
            example["class_images"]        = cls_pv
            example["class_masks"]         = cls_mask
            example["class_masked_images"] = cls_pv * (cls_mask < 0.5)
            example["class_prompt"]        = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation: bool = False):
    pixel_values   = [e["pixel_values"]  for e in examples]
    masks          = [e["masks"]         for e in examples]
    masked_images  = [e["masked_images"] for e in examples]
    depth_values   = [e["depth_values"]  for e in examples]
    input_ids_one  = [e["input_ids_one"] for e in examples]
    input_ids_two  = [e["input_ids_two"] for e in examples]
    original_sizes = [e["original_size"] for e in examples]
    crop_top_lefts = [e["crop_top_left"] for e in examples]
    stems          = [e["stem"]          for e in examples]
    latent_cached  = all(e.get("latent_cached", False) for e in examples)

    if with_prior_preservation:
        pixel_values   += [e["class_images"]        for e in examples]
        masks          += [e["class_masks"]          for e in examples]
        masked_images  += [e["class_masked_images"]  for e in examples]
        depth_values   += [torch.zeros_like(e["depth_values"]) for e in examples]  # no depth for class imgs
        input_ids_one  += [e["input_ids_one"]        for e in examples]
        input_ids_two  += [e["input_ids_two"]        for e in examples]
        original_sizes += [e["original_size"]        for e in examples]
        crop_top_lefts += [e["crop_top_left"]        for e in examples]
        stems          += [e["stem"]                 for e in examples]

    return {
        "pixel_values":   torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
        "masks":          torch.stack(masks).float(),
        "masked_images":  torch.stack(masked_images).to(memory_format=torch.contiguous_format).float(),
        "depth_values":   torch.stack(depth_values).to(memory_format=torch.contiguous_format).float(),
        "input_ids_one":  torch.stack(input_ids_one),
        "input_ids_two":  torch.stack(input_ids_two),
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "stem":           stems,
        "latent_cached":  latent_cached,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prior-preservation class image generation
# ──────────────────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


def generate_class_images(cfg, accelerator):
    class_images_dir = Path(cfg.dreambooth.class_data_dir)
    class_images_dir.mkdir(parents=True, exist_ok=True)
    cur    = len(list(class_images_dir.iterdir()))
    target = cfg.dreambooth.num_class_images

    if cur >= target:
        return

    logger.info(f"Generating {target - cur} class images in {class_images_dir} …")

    has_fp16    = torch.cuda.is_available() or torch.backends.mps.is_available()
    torch_dtype = torch.float16 if has_fp16 else torch.float32

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)

    res        = cfg.training.resolution
    dummy_img  = Image.fromarray(np.full((res, res, 3), 255, dtype=np.uint8))
    mask_arr   = np.zeros((res, res), dtype=np.uint8)
    mask_arr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
    dummy_mask = Image.fromarray(mask_arr)

    ds = PromptDataset(cfg.dreambooth.class_prompt, target - cur)
    dl = accelerator.prepare(DataLoader(ds, batch_size=cfg.training.sample_batch_size))

    for batch in tqdm(dl, desc="Generating class images",
                      disable=not accelerator.is_local_main_process):
        images = pipeline(
            prompt=list(batch["prompt"]),
            image=[dummy_img] * len(batch["prompt"]),
            mask_image=[dummy_mask] * len(batch["prompt"]),
            height=res, width=res,
        ).images
        for i, img in enumerate(images):
            h = insecure_hashlib.sha1(img.tobytes()).hexdigest()
            img.save(class_images_dir / f"{batch['index'][i].item() + cur}-{h}.jpg")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# ★ NEW: Pre-compute VAE latents (Fix 2)
# ──────────────────────────────────────────────────────────────────────────────

def precompute_latents(train_dataset, vae, accelerator, weight_dtype, batch_size=8):
    """
    Pre-encode all training images to VAE latents 1 time, cache in dataset.
    Eliminates per-step VAE encoding bottleneck.
    
    Memory estimate: ~2.6 GB per 1000 images at 1024px (4-channel latents in bfloat16)
    For 2900 images: ~7.5 GB
    """
    logger.info(f"★ Pre-computing VAE latents for {len(train_dataset)} images …")
    
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No workers to avoid duplicate encoding
        pin_memory=True,
    )
    
    vae = vae.to(accelerator.device)
    vae.eval()
    
    total_encoded = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="VAE encoding", position=0, leave=True)):
            # Move to device and ensure float32 for VAE
            pv = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
            mi = batch["masked_images"].to(accelerator.device, dtype=torch.float32)
            stems = batch["stem"]  # List of stems for this batch
            
            # Encode to latents
            with torch.autocast(accelerator.device.type, dtype=weight_dtype, enabled=True):
                lat = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
                mlat = vae.encode(mi).latent_dist.sample() * vae.config.scaling_factor
            
            # Move to CPU as bfloat16 to save VRAM
            lat = lat.to("cpu", dtype=weight_dtype)
            mlat = mlat.to("cpu", dtype=weight_dtype)
            
            # Store in dataset cache by stem
            for i, stem in enumerate(stems):
                train_dataset._latent_cache[stem] = (lat[i].clone(), mlat[i].clone())
                total_encoded += 1
    
    logger.info(f"✓ Latent cache complete: {total_encoded} latents cached (~{total_encoded * 4.5 / 1000:.1f} GB)")
    
    # Move VAE back to CPU to free VRAM
    vae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# WandB setup
# ──────────────────────────────────────────────────────────────────────────────

def init_wandb(cfg, accelerator):
    if not is_wandb_available():
        raise ImportError("wandb is not installed. Run: pip install wandb")
    if not accelerator.is_main_process:
        return

    api_key = getattr(cfg.logging, "wandb_api_key", None)
    if api_key:
        wandb.login(key=api_key)

    wandb.init(
        project=getattr(cfg.logging, "wandb_project", "sdxl-inpaint-dreambooth"),
        entity=getattr(cfg.logging, "wandb_entity", None),
        name=getattr(cfg.logging, "wandb_run_name", None) or cfg.logging.run_name,
        tags=list(getattr(cfg.logging, "wandb_tags", []) or []),
        notes=getattr(cfg.logging, "wandb_notes", "") or "",
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info(f"WandB run initialised: {wandb.run.name}  ({wandb.run.url})")


def log_wandb_images(images: list, prompts: list, step: int, tag: str = "validation"):
    if not is_wandb_available() or wandb.run is None:
        return
    wandb.log(
        {tag: [wandb.Image(img, caption=p) for img, p in zip(images, prompts)]},
        step=step,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_import_metrics():
    lpips_fn = ssim_fn = psnr_fn = None

    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").eval()
        logger.info("LPIPS loaded (alex net).")
    except ImportError:
        logger.warning("lpips not installed — skipping LPIPS. Run: pip install lpips")

    try:
        from torchmetrics.functional.image import (
            structural_similarity_index_measure as _ssim,
            peak_signal_noise_ratio as _psnr,
        )
        ssim_fn = _ssim
        psnr_fn = _psnr
        logger.info("torchmetrics loaded (SSIM + PSNR).")
    except ImportError:
        logger.warning("torchmetrics not installed — skipping SSIM/PSNR. Run: pip install torchmetrics")

    return lpips_fn, ssim_fn, psnr_fn


def _pil_to_chw(pil_img: Image.Image, device) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _pil_to_01(pil_img: Image.Image, device) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_metrics(pred, gt, mask, device, lpips_fn=None, ssim_fn=None, psnr_fn=None):
    results  = {}
    mask_np  = (np.array(mask.convert("L")) > 127)
    if not mask_np.any():
        return results

    ys, xs = np.where(mask_np)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pred_crop = pred.crop((x1, y1, x2, y2))
    gt_crop   = gt.crop((x1, y1, x2, y2))

    if lpips_fn is not None:
        try:
            lpips_fn = lpips_fn.to(device)
            with torch.no_grad():
                lp = lpips_fn(_pil_to_chw(pred_crop, device),
                              _pil_to_chw(gt_crop,   device)).item()
            results["val/lpips"] = lp
        except Exception as e:
            logger.warning(f"LPIPS error: {e}")

    if ssim_fn is not None:
        try:
            with torch.no_grad():
                sv = ssim_fn(_pil_to_01(pred_crop, device),
                             _pil_to_01(gt_crop,   device), data_range=1.0).item()
            results["val/ssim"] = sv
        except Exception as e:
            logger.warning(f"SSIM error: {e}")

    if psnr_fn is not None:
        try:
            with torch.no_grad():
                pv = psnr_fn(_pil_to_01(pred_crop, device),
                             _pil_to_01(gt_crop,   device), data_range=1.0).item()
            results["val/psnr"] = pv
        except Exception as e:
            logger.warning(f"PSNR error: {e}")

    return results


# ★ MODIFIED: 4-panel comparison image
def make_comparison_image(
    original:     Image.Image,
    mask:         Image.Image,
    result:       Image.Image,
    mask_dilated: Image.Image | None = None,
) -> Image.Image:
    """
    Stitch 4 images side-by-side:

        [Original] | [Mask overlay (red tint)] | [Erased / hole] | [Inpainted result]

    Panel 3 ("Erased / hole") shows the original image with the masked region
    replaced by mid-gray, making the "hole" visible at a glance.

    If mask_dilated is provided it is used for the overlay and erased panels
    (so the viewer sees the actual region passed to the pipeline).
    """
    w, h     = original.size
    orig_arr = np.array(original.convert("RGB"))

    # Use dilated mask for display if available, else original mask
    display_mask = mask_dilated if mask_dilated is not None else mask
    mask_arr     = np.array(display_mask.convert("L"))

    # ── Panel 2: red overlay ──────────────────────────────────────────────
    overlay = orig_arr.copy()
    overlay[mask_arr > 127] = [255, 80, 80]
    overlay_img = Image.fromarray(
        (orig_arr * 0.5 + overlay * 0.5).astype(np.uint8)
    )

    # ── Panel 3: erased hole ──────────────────────────────────────────────
    # Masked pixels → neutral gray (128, 128, 128) so the "missing" region
    # is clearly visible without colour bias.
    erased     = orig_arr.copy()
    erased[mask_arr > 127] = [180, 180, 180]
    erased_img = Image.fromarray(erased)

    # ── Stitch ────────────────────────────────────────────────────────────
    combined = Image.new("RGB", (w * 4, h))
    combined.paste(original,    (0,     0))
    combined.paste(overlay_img, (w,     0))
    combined.paste(erased_img,  (w * 2, 0))
    combined.paste(result,      (w * 3, 0))
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# ★ NEW: Feather blend helper
# ──────────────────────────────────────────────────────────────────────────────

def feather_blend(
    original: Image.Image,
    inpainted: Image.Image,
    mask: Image.Image,
    feather_radius: int = 15,
) -> Image.Image:
    """
    Blend inpainted result with original at mask boundary.
    Feathered mask = gaussian blur on binary mask → smooth gradient
    at edges → color transitions gradually from original to inpainted.

    Args:
        original:       PIL RGB image
        inpainted:      PIL RGB image (same size)
        mask:           PIL L image (0=keep, 255=inpainted)
        feather_radius: blur radius in pixels (larger = softer transition)
    """
    from PIL import ImageFilter

    # Gaussian blur mask → feathered edges
    feathered = mask.convert("L").filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )

    # Normalize to [0, 1]
    f_arr = np.array(feathered).astype(np.float32) / 255.0
    o_arr = np.array(original.convert("RGB")).astype(np.float32)
    i_arr = np.array(inpainted.convert("RGB")).astype(np.float32)

    # Blend: original * (1 - alpha) + inpainted * alpha
    f_mask = f_arr[:, :, np.newaxis]  # (H, W, 1)
    blended = o_arr * (1.0 - f_mask) + i_arr * f_mask

    return Image.fromarray(blended.astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

_METRICS_CACHE: tuple | None = None

# ★ DISABLED: Early stopping state (tracks best validation LPIPS)
# _BEST_VAL_LPIPS: float = float("inf")
# _PATIENCE_COUNTER: int = 0


# ★ MODIFIED: mask dilation + 4-panel comparison
def log_validation(
    pipeline, cfg, accelerator, epoch, step, weight_dtype,
    val_paths:      list[Path] | None        = None,
    val_masks_root: Path | None              = None,
    val_depths_root: Path | None             = None,
    val_prompt_map: dict[str, str] | None    = None,
):
    if not val_paths:
        logger.info("No val split — skipping validation.")
        return []

    logger.info(
        f"Running validation (epoch {epoch}, step {step}, "
        f"{len(val_paths)} val images) …"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
        if cfg.training.seed is not None else None
    )

    res     = cfg.training.resolution
    val_dir = Path(cfg.training.output_dir).resolve() / "validation"  # ★ Use absolute path
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # ★ Ensure all parent directories exist (fix for multi-GPU)
    if not val_dir.exists():
        val_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created validation directory: {val_dir}")

    autocast_ctx = (
        torch.autocast(accelerator.device.type, dtype=weight_dtype)
        if torch.cuda.is_available() else nullcontext()
    )

    global _METRICS_CACHE
    if _METRICS_CACHE is None:
        _METRICS_CACHE = _try_import_metrics()
    lpips_fn, ssim_fn, psnr_fn = _METRICS_CACHE

    # ── Mask dilation config ──────────────────────────────────────────────
    # dilation_pct : 10 % of min(H,W) radius expansion
    # max_area_pct : hard cap — dilated mask never exceeds 70 % of image
    mask_dilation_pct = float(getattr(cfg.validation, "mask_dilation_pct", 0.10))
    mask_max_area_pct = float(getattr(cfg.validation, "mask_max_area_pct", 0.70))

    num_to_show   = min(cfg.validation.num_validation_images, len(val_paths))
    all_metrics:  list[dict] = []
    wandb_images: list       = []
    wandb_prompts: list      = []

    num_to_validate = min(num_to_show, len(val_paths))  # chỉ 4-8 ảnh
    val_subset = random.sample(val_paths, num_to_validate)  # ★ Random selection each validation run
    for idx, img_path in enumerate(tqdm(val_subset, desc="Validation", leave=False)):
        # ── Load original ─────────────────────────────────────────────────
        # ★ SPEED: Use fast image loader
        pil_orig = open_image_fast(str(img_path), mode="RGB") or Image.open(img_path).convert("RGB")
        pil_orig = pil_orig.resize((res, res), Image.LANCZOS)

        # ── Load mask ─────────────────────────────────────────────────────
        pil_mask_raw = None
        if val_masks_root is not None:
            for ext in [img_path.suffix, ".png", ".jpg", ".jpeg"]:
                mp = val_masks_root / (img_path.stem + ext)
                if mp.exists():
                    # ★ SPEED: Use fast loader
                    m_img = open_image_fast(str(mp), mode="L") or Image.open(mp).convert("L")
                    m_img = m_img.resize((res, res), Image.NEAREST)
                    marr = np.array(m_img)
                    if marr.max() > 0:
                        pil_mask_raw = Image.fromarray(marr)
                    break

        if pil_mask_raw is None:
            marr = np.zeros((res, res), dtype=np.uint8)
            marr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
            pil_mask_raw = Image.fromarray(marr)

        # ── Load depth ────────────────────────────────────────────────────
        pil_depth = None
        if val_depths_root is not None:
            for ext in [img_path.suffix, ".png", ".jpg", ".jpeg", ".webp"]:
                dp = val_depths_root / (img_path.stem + ext)
                if dp.exists():
                    # ★ SPEED: Use fast loader
                    pil_depth = open_image_fast(str(dp), mode="RGB") or Image.open(dp).convert("RGB")
                    pil_depth = pil_depth.resize((res, res), Image.LANCZOS)
                    break
        if pil_depth is None:
            pil_depth = Image.new("RGB", (res, res), (0, 0, 0))

        # ★ NEW: Dilate mask by 10%, cap at 70% of image area ─────────────
        raw_arr    = np.array(pil_mask_raw.convert("L"))
        dilated_arr = dilate_mask_pct(
            raw_arr,
            dilation_pct=mask_dilation_pct,
            max_area_pct=mask_max_area_pct,
        )
        pil_mask_dilated = Image.fromarray(dilated_arr)

        # ★ SPEED/QUALITY: Apply Gaussian Blur to the mask so SDXL's VAE blends edges smoothly
        blur_radius = float(getattr(cfg.validation, "mask_blur_radius", int(res * 0.02)))
        pil_mask_blurred = pil_mask_dilated.filter(ImageFilter.GaussianBlur(blur_radius))

        # Log dilation stats on first image of first run
        if idx == 0:
            orig_pct  = raw_arr.astype(bool).mean() * 100
            dil_pct   = dilated_arr.astype(bool).mean() * 100
            logger.info(
                f"  Mask dilation: {orig_pct:.1f}% → {dil_pct:.1f}%  "
                f"(cap={mask_max_area_pct*100:.0f}%)"
            )

        # ── Prompt ────────────────────────────────────────────────────────
        prompt = (val_prompt_map or {}).get(img_path.name, "interior design, high quality")

        # ── Inference parameters with quality boost ───────────────────────
        num_steps = getattr(cfg.validation, "num_inference_steps", 50)
        guidance_scale = float(getattr(cfg.validation, "guidance_scale", 12.0))
        strength = float(getattr(cfg.validation, "strength", 1.0))
        controlnet_cond_scale = float(getattr(cfg.validation, "controlnet_conditioning_scale", 0.7))
        
        if idx == 0:
            logger.info(
                f"Validation inference: steps={num_steps}, guidance={guidance_scale}, "
                f"strength={strength}, controlnet_scale={controlnet_cond_scale}"
            )

        # ── Inpaint with DILATED AND BLURRED mask, DEPTH, and quality boost ──────────
        with autocast_ctx:
            out = pipeline(
                prompt=prompt,
                image=pil_orig,
                mask_image=pil_mask_blurred,   # ★ use dilated & blurred mask
                control_image=pil_depth,       # ★ depth conditioning
                height=res, width=res,
                num_inference_steps=num_steps,  # ★ 50 steps (was 25)
                guidance_scale=guidance_scale,  # ★ 12.0 for interior design
                strength=strength,              # ★ 1.0 = high fidelity
                controlnet_conditioning_scale=controlnet_cond_scale,  # ★ 0.7 (tamer depth)
                generator=generator,
            ).images[0]

        # ★ NEW: Feather blend — smooth color transition at mask boundary
        # (This is a second-pass pixel blend using the blurred mask to ensure 
        # perfect consistency with the unmasked original regions)
        out = feather_blend(
            original=pil_orig,
            inpainted=out,
            mask=pil_mask_blurred,
            feather_radius=0,  # We already blurred the mask, so feather_blend just applies the alpha
        )

        out.save(val_dir / f"epoch{epoch:04d}_step{step:07d}_{img_path.stem}.png")

        # ── Metrics (raw mask for fair comparison) ────────────────────────
        m = compute_metrics(
            pred=out, gt=pil_orig, mask=pil_mask_raw,
            device=accelerator.device,
            lpips_fn=lpips_fn, ssim_fn=ssim_fn, psnr_fn=psnr_fn,
        )
        all_metrics.append(m)

        # ── Collect for WandB (first N images only) ───────────────────────
        if idx < num_to_show:
            # ★ 4-panel comparison: original | overlay | erased | inpainted
            comparison = make_comparison_image(
                original=pil_orig,
                mask=pil_mask_raw,
                result=out,
                mask_dilated=pil_mask_dilated,  # show dilated mask in panels 2+3
            )
            comparison.save(
                val_dir / f"epoch{epoch:04d}_step{step:07d}_{img_path.stem}_cmp.png"
            )
            wandb_images.append(comparison)
            wandb_prompts.append(
                f"[orig | mask(+dil) | erased hole | inpaint]  {prompt}"
            )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    agg = {}
    for key in ["val/lpips", "val/ssim", "val/psnr"]:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            agg[key]           = float(np.mean(vals))
            agg[key + "_std"]  = float(np.std(vals))
            agg[key + "_best"] = float(np.min(vals) if "lpips" in key else np.max(vals))

    if agg:
        logger.info(
            "  val metrics → "
            + "  ".join(
                f"{k}={v:.4f}"
                for k, v in agg.items()
                if "_std" not in k and "_best" not in k
            )
        )

    if accelerator.is_main_process and is_wandb_available() and wandb.run is not None:
        log_dict = {**agg, "val/epoch": epoch}
        log_dict["validation"] = [
            wandb.Image(img, caption=p)
            for img, p in zip(wandb_images, wandb_prompts)
        ]
        wandb.log(log_dict, step=step)

    # ★ NEW: Return both images and metrics for early stopping check
    return wandb_images, agg


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load hooks
# ──────────────────────────────────────────────────────────────────────────────

def build_save_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg):
    def save_model_hook(models, weights, output_dir):
        # NOTE: With ZeRO-3, GatheredParameters is a collective op — ALL ranks
        # must enter.  We do NOT early-return on non-main processes before gather.
        unet_lora_layers = te1_lora_layers = te2_lora_layers = None

        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            # Gather partitioned params (collective); only rank-0 reads data
            with _zero3_gather(unwrapped, accelerator, modifier_rank=0):
                if accelerator.is_main_process:
                    if isinstance(unwrapped, type(accelerator.unwrap_model(unet))):
                        unet_lora_layers = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped)
                        )
                    elif isinstance(unwrapped, type(accelerator.unwrap_model(text_encoder_one))):
                        te1_lora_layers = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped)
                        )
                    elif text_encoder_two is not None and isinstance(
                        unwrapped, type(accelerator.unwrap_model(text_encoder_two))
                    ):
                        te2_lora_layers = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped)
                        )
            if weights:            
                weights.pop()

        # ★ FIX: Save LoRA weights using safetensors directly.
        # StableDiffusionXLInpaintPipeline.save_lora_weights() crashes due to
        # API changes across diffusers versions. Direct save_file() is robust.
        if accelerator.is_main_process:
            try:
                os.makedirs(output_dir, exist_ok=True)

                # ★ FIX: Normalize any remaining PEFT-format keys (.lora_A/.lora_B)
                # to diffusers format (.lora.down/.lora.up). convert_state_dict_to_diffusers
                # may miss ff layers in some PEFT/diffusers version combos.
                def _normalize_lora_keys(sd):
                    if sd is None:
                        return sd
                    return {
                        k.replace(".lora_A.weight", ".lora.down.weight")
                         .replace(".lora_B.weight", ".lora.up.weight"): v
                        for k, v in sd.items()
                    }
                unet_lora_layers = _normalize_lora_keys(unet_lora_layers)
                te1_lora_layers  = _normalize_lora_keys(te1_lora_layers)
                te2_lora_layers  = _normalize_lora_keys(te2_lora_layers)

                # Merge all LoRA state dicts with standard prefixes
                all_lora_weights = {}
                if unet_lora_layers:
                    for k, v in unet_lora_layers.items():
                        all_lora_weights[f"unet.{k}"] = v
                if te1_lora_layers:
                    for k, v in te1_lora_layers.items():
                        all_lora_weights[f"text_encoder.{k}"] = v
                if te2_lora_layers:
                    for k, v in te2_lora_layers.items():
                        all_lora_weights[f"text_encoder_2.{k}"] = v

                save_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
                save_file(all_lora_weights, save_path)
                logger.info(
                    f"  LoRA checkpoint saved ({len(all_lora_weights)} tensors → {save_path})"
                )
            except Exception as e:
                logger.error(f"  ⚠️ Failed to save LoRA weights: {e}")

    return save_model_hook


def build_load_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg):
    def load_model_hook(models, input_dir):
        unet_ = te1_ = te2_ = None
        while models:
            m = models.pop()
            unwrapped = accelerator.unwrap_model(m)
            if isinstance(unwrapped, type(accelerator.unwrap_model(unet))):
                unet_ = m
            elif isinstance(unwrapped, type(accelerator.unwrap_model(text_encoder_one))):
                te1_ = m
            elif text_encoder_two is not None and isinstance(
                unwrapped, type(accelerator.unwrap_model(text_encoder_two))
            ):
                te2_ = m
            else:
                raise ValueError(f"Unexpected model type: {m.__class__}")

        lora_sd, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_sd    = {k.replace("unet.", ""): v for k, v in lora_sd.items() if k.startswith("unet.")}
        unet_sd    = convert_unet_state_dict_to_peft(unet_sd)

        unet_unwrapped = accelerator.unwrap_model(unet_)
        # All ranks gather for ZeRO-3 (collective); modifier_rank=None = all can write
        with _zero3_gather(unet_unwrapped, accelerator, modifier_rank=None):
            incompatible = set_peft_model_state_dict(unet_unwrapped, unet_sd, adapter_name="default")
            if incompatible and getattr(incompatible, "unexpected_keys", None):
                logger.warning(f"Unexpected keys when loading LoRA: {incompatible.unexpected_keys}")

        if cfg.dreambooth.train_text_encoder:
            te1_unwrapped = accelerator.unwrap_model(te1_)
            with _zero3_gather(te1_unwrapped, accelerator, modifier_rank=None):
                _set_state_dict_into_text_encoder(lora_sd, prefix="text_encoder.", text_encoder=te1_unwrapped)
            if te2_ is not None:
                te2_unwrapped = accelerator.unwrap_model(te2_)
                with _zero3_gather(te2_unwrapped, accelerator, modifier_rank=None):
                    _set_state_dict_into_text_encoder(lora_sd, prefix="text_encoder_2.", text_encoder=te2_unwrapped)

        if cfg.training.mixed_precision == "fp16":
            ms = [unet_]
            if cfg.dreambooth.train_text_encoder:
                ms += [te1_, te2_]
            cast_training_params([m for m in ms if m is not None])

    return load_model_hook


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)
    
    # ⚠️  If running via 'accelerate launch', RANK env var is set
    #     → DO NOT override CUDA_VISIBLE_DEVICES; let accelerate manage it
    # Otherwise (single GPU mode via python directly):
    #     → CUDA_VISIBLE_DEVICES is set in __main__ block
    is_distributed = 'RANK' in os.environ
    num_visible_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    if is_distributed:
        logger.info(f"[Distributed] Using {num_visible_gpus} GPU(s) (accelerate managed)")
    else:
        logger.info(f"[Single GPU] Using 1 GPU — CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

    # ── Memory limits (VRAM 93%, RAM reserve 2GB) ─────────────────────────
    set_memory_limits(
        vram_fraction=getattr(cfg, "vram_fraction", 0.93),
        ram_reserve_gb=getattr(cfg, "ram_reserve_gb", 2.0),
    )

    # ── CPU thread limit directly in Python (OMP_NUM_THREADS/MKL_NUM_THREADS already set in bash) ──
    torch.set_num_threads(2)
    logger.info("[CPU] PyTorch CPU threads set to 2 (backup for OMP_NUM_THREADS/MKL_NUM_THREADS)")

    # ── Start RAM monitoring daemon (prevents OOM killer from killing random processes) ──────────
    start_ram_monitor_daemon(threshold_pct=90.0, high_count_limit=1, check_interval=5.0)

    # ── Accelerator ───────────────────────────────────────────────────────
    project_config = ProjectConfiguration(
        project_dir=cfg.training.output_dir,
        logging_dir=os.path.join(cfg.training.output_dir, "logs"),
    )
    kwargs      = DistributedDataParallelKwargs(find_unused_parameters=True)
    report_to   = cfg.logging.report_to if cfg.logging.report_to != "none" else None
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=report_to,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if accelerator.is_main_process:
        os.makedirs(cfg.training.output_dir, exist_ok=True)

    set_seed(cfg.training.seed)

    use_wandb = cfg.logging.report_to == "wandb"
    if use_wandb and accelerator.is_main_process:
        init_wandb(cfg, accelerator)

    if cfg.dreambooth.with_prior_preservation:
        if accelerator.is_main_process:
            generate_class_images(cfg, accelerator)
        accelerator.wait_for_everyone()

    # ── Models ────────────────────────────────────────────────────────────
    pretrained = cfg.model.pretrained_model_name_or_path
    
    # Determine weight_dtype early (before loading models for low_cpu_mem_usage)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer_one = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer_2", use_fast=False)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(pretrained, None, "text_encoder_2")

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    do_edm = getattr(cfg.training, "do_edm_style_training", False)
    if do_edm:
        try:
            noise_scheduler = EDMEulerScheduler.from_pretrained(pretrained, subfolder="scheduler")
        except Exception:
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
        logger.info("EDM-style training enabled.")

    # ── Stagger model loading to avoid 4× RAM spike ──────────────────────
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    vae_path = getattr(cfg.model, "pretrained_vae_model_name_or_path", None) or pretrained
    
    local_rank = accelerator.local_process_index
    num_local = accelerator.num_processes
    for i in range(num_local):
        if i == local_rank:
            logger.info(f"[Rank {local_rank}] Loading models...")
            
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                pretrained, subfolder="text_encoder", low_cpu_mem_usage=True, torch_dtype=weight_dtype
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                pretrained, subfolder="text_encoder_2", low_cpu_mem_usage=True, torch_dtype=weight_dtype
            )
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if not getattr(cfg.model, "pretrained_vae_model_name_or_path", None) else None,
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
            unet = UNet2DConditionModel.from_pretrained(
                pretrained, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=weight_dtype
            )
            
            controlnet_path = getattr(cfg.model, "pretrained_controlnet_model_name_or_path", "diffusers/controlnet-depth-sdxl-1.0")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path, low_cpu_mem_usage=True, torch_dtype=weight_dtype
            )
            
            # Move to GPU immediately and free CPU
            vae.requires_grad_(False)
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            unet.requires_grad_(False)
            controlnet.requires_grad_(False)
            vae.eval()
            controlnet.eval()
            
            # With DeepSpeed ZeRO-3, don't move models to device before prepare() —
            # DeepSpeed handles parameter placement & partitioning.
            # Frozen models NOT passed to prepare() must be moved manually.
            # (ControlNet is frozen, so move it regardless of DeepSpeed)
            if not is_deepspeed:
                unet.to(accelerator.device, dtype=weight_dtype)
            
            controlnet.to(accelerator.device, dtype=weight_dtype)
            vae.to(accelerator.device, dtype=torch.float32)
            vae.enable_slicing()  # ★ NEW: Reduce peak VRAM during VAE decode
            text_encoder_one.to(accelerator.device, dtype=weight_dtype)
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)
            
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"[Rank {local_rank}] Models loaded and moved to GPU, CPU freed")
        
        # All ranks wait here before next rank loads
        accelerator.wait_for_everyone()

    if getattr(cfg.training, "enable_xformers", False):
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers enabled.")
        else:
            logger.warning("xFormers requested but not available.")

    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── LoRA / DoRA ───────────────────────────────────────────────────────
    rank     = cfg.lora.rank
    alpha    = getattr(cfg.lora, "alpha", rank)
    dropout  = getattr(cfg.lora, "dropout", 0.0)
    use_dora = getattr(cfg.lora, "use_dora", False)

    rank_attn  = rank
    rank_ff    = getattr(cfg.lora, "rank_ff",    rank // 2)
    rank_xattn = getattr(cfg.lora, "rank_xattn", rank // 2)

    # ── Build consolidated target_modules list ──
    all_targets = ["to_k", "to_q", "to_v", "to_out.0"]  # Self-attention

    # Detect cross-attention modules
    _cross_modules   = ["add_k_proj", "add_q_proj", "add_v_proj", "add_out_proj"]
    _cross_available = []
    for name, _ in unet.named_modules():
        for t in _cross_modules:
            if t in name and t not in _cross_available:
                _cross_available.append(t)
    
    all_targets += _cross_available

    # Feed-forward modules
    ff_targets = ["ff.net.0.proj", "ff.net.2"]
    all_targets += ff_targets

    # Build rank_pattern and alpha_pattern for different module groups
    rank_pattern = {}
    alpha_pattern = {}
    
    # Feed-forward: use rank_ff
    for t in ff_targets:
        rank_pattern[t] = rank_ff
        alpha_pattern[t] = rank_ff
    
    # Cross-attention: use rank_xattn
    for t in _cross_available:
        rank_pattern[t] = rank_xattn
        alpha_pattern[t] = rank_xattn
    
    # Create SINGLE LoraConfig with all targets (self-attn uses default rank)
    unet_lora_config = LoraConfig(
        r=rank_attn,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=all_targets,
        rank_pattern=rank_pattern,      # Override rank for ff and cross-attn
        alpha_pattern=alpha_pattern,    # Override alpha for ff and cross-attn
        **({"use_dora": True} if use_dora else {}),
    )
    
    # Add adapter ONCE (replaces any existing adapters)
    unet.add_adapter(unet_lora_config)
    
    logger.info(f"LoRA targets: {all_targets}")
    logger.info(f"Rank: attn={rank_attn}, xattn={rank_xattn}, ff={rank_ff}")
    if _cross_available:
        logger.info(f"Cross-attention modules found: {_cross_available}")
    else:
        logger.warning("Cross-attention modules not found.")

    # Verify trainable params
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"UNet LoRA params: {trainable:,} trainable / {total_p:,} total "
        f"({100 * trainable / total_p:.3f}%) [should be 20-40M for SDXL rank=32]"
    )

    train_te = cfg.dreambooth.train_text_encoder
    if train_te:
        te_lora_cfg = get_lora_config(
            rank=rank, alpha=alpha, dropout=dropout, use_dora=use_dora,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(te_lora_cfg)
        text_encoder_two.add_adapter(te_lora_cfg)
        if cfg.training.gradient_checkpointing:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    if cfg.training.mixed_precision == "fp16":
        models_to_cast = [unet]
        if train_te:
            models_to_cast += [text_encoder_one, text_encoder_two]
        cast_training_params(models_to_cast, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(
        build_save_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg)
    )
    accelerator.register_load_state_pre_hook(
        build_load_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg)
    )

    # ── ★ NEW: Auxiliary loss setup ───────────────────────────────────────
    loss_weights = getattr(cfg, "loss_weights", None)
    logger.info(f"[DEBUG] cfg.loss_weights = {loss_weights}")  # ← Debug print

    # Build a loss weights namespace with proper __init__ to read from config
    class _LW:
        """Loss weights namespace with defaults and config reading."""
        def __init__(self, loss_weights_config=None):
            # Set defaults
            self.diffusion_weight = 1.0
            self.pixel_weight = 0.0
            self.perceptual_weight = 0.0
            self.clip_weight = 0.0
            self.boundary_weight = 0.0
            self.depth_weight = 0.0
            self.semantic_weight = 0.0
            self.aux_loss_every_n_steps = 1
            self.aux_resolution_scale = 1.0
            
            # Override with config values if provided
            if loss_weights_config is not None:
                for attr in [
                    "diffusion_weight", "pixel_weight", "perceptual_weight",
                    "clip_weight", "boundary_weight", "depth_weight", "semantic_weight",
                    "aux_loss_every_n_steps", "aux_resolution_scale"
                ]:
                    val = getattr(loss_weights_config, attr, None)
                    if val is not None:
                        if attr == "aux_loss_every_n_steps":
                            setattr(self, attr, int(val))
                        else:
                            setattr(self, attr, float(val))

    lw = _LW(loss_weights)

    # Warn if weights don't sum close to 1
    total_w = (
        lw.diffusion_weight + lw.pixel_weight + lw.perceptual_weight +
        lw.clip_weight + lw.boundary_weight + lw.depth_weight + lw.semantic_weight
    )
    if abs(total_w - 1.0) > 0.01:
        logger.warning(
            f"loss_weights sum to {total_w:.4f} (expected ~1.0). "
            "Consider re-normalising in your YAML."
        )

    # ★ DEBUG: Print loaded weights
    logger.info(
        f"[Loaded loss_weights] diffusion={lw.diffusion_weight:.2f}, "
        f"pixel={lw.pixel_weight:.2f}, perceptual={lw.perceptual_weight:.2f}, "
        f"clip={lw.clip_weight:.2f}, boundary={lw.boundary_weight:.2f}, "
        f"depth={lw.depth_weight:.2f}, semantic={lw.semantic_weight:.2f}, "
        f"aux_loss_every_n_steps={lw.aux_loss_every_n_steps}"
    )

    has_aux = any([
        lw.pixel_weight > 0, lw.perceptual_weight > 0, lw.clip_weight > 0,
        lw.boundary_weight > 0, lw.depth_weight > 0, lw.semantic_weight > 0,
    ])
    aux_computer = (
        AuxiliaryLossComputer(accelerator.device, weight_dtype) if has_aux else None
    )
    if has_aux:
        logger.info(
            f"Auxiliary losses enabled  "
            f"(diffusion={lw.diffusion_weight:.2f} "
            f"pixel={lw.pixel_weight:.2f} "
            f"perceptual={lw.perceptual_weight:.2f} "
            f"clip={lw.clip_weight:.2f} "
            f"boundary={lw.boundary_weight:.2f} "
            f"depth={lw.depth_weight:.2f} "
            f"semantic={lw.semantic_weight:.2f}  "
            f"every={lw.aux_loss_every_n_steps} steps)"
        )

    # ── Optimiser ─────────────────────────────────────────────────────────
    unet_params        = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize = [{"params": unet_params, "lr": cfg.training.learning_rate}]

    if train_te:
        te_lr = getattr(cfg.training, "text_encoder_lr", cfg.training.learning_rate)
        params_to_optimize += [
            {"params": list(filter(lambda p: p.requires_grad, text_encoder_one.parameters())),
             "lr": te_lr, "weight_decay": cfg.training.adam_weight_decay},
            {"params": list(filter(lambda p: p.requires_grad, text_encoder_two.parameters())),
             "lr": te_lr, "weight_decay": cfg.training.adam_weight_decay},
        ]

    optimizer_name = getattr(cfg.training, "optimizer", "adamw").lower()

    if optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("Install prodigyopt: pip install prodigyopt")
        optimizer = prodigyopt.Prodigy(
            params_to_optimize,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            weight_decay=cfg.training.adam_weight_decay,
            eps=cfg.training.adam_epsilon,
            decouple=getattr(cfg.training, "prodigy_decouple", True),
            use_bias_correction=getattr(cfg.training, "prodigy_use_bias_correction", True),
            safeguard_warmup=getattr(cfg.training, "prodigy_safeguard_warmup", True),
        )
    else:
        if getattr(cfg.training, "use_8bit_adam", False):
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                logger.warning("bitsandbytes not available – using standard AdamW.")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            params_to_optimize,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            weight_decay=cfg.training.adam_weight_decay,
            eps=cfg.training.adam_epsilon,
        )

    # ── Data ──────────────────────────────────────────────────────────────
    dataset_dir  = Path(cfg.data.dataset_dir)
    images_dir   = str(dataset_dir / getattr(cfg.data, "images_subdir",   "images"))
    masks_dir    = str(dataset_dir / getattr(cfg.data, "masks_subdir",    "masks"))
    depths_dir   = str(dataset_dir / getattr(cfg.data, "depths_subdir",   "depths"))
    captions_dir = str(dataset_dir / getattr(cfg.data, "captions_subdir", "captions"))

    logger.info(
        f"Dataset dirs:\n"
        f"  images   = {images_dir}\n"
        f"  masks    = {masks_dir}\n"
        f"  depths   = {depths_dir}\n"
        f"  captions = {captions_dir}"
    )

    splits = make_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        depths_dir=depths_dir,
        captions_dir=captions_dir,
        train_ratio=getattr(cfg.data, "train_ratio", 0.8),
        val_ratio=getattr(cfg.data,   "val_ratio",   0.1),
        seed=cfg.training.seed,
    )

    train_dataset = DreamBoothInpaintingDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        depths_dir=depths_dir,
        captions_dir=captions_dir,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        file_list=splits["train"],
        fallback_caption=getattr(cfg.dreambooth, "instance_prompt", ""),
        class_prompt=cfg.dreambooth.class_prompt if cfg.dreambooth.with_prior_preservation else None,
        class_data_dir=cfg.dreambooth.class_data_dir if cfg.dreambooth.with_prior_preservation else None,
        class_num=cfg.dreambooth.num_class_images,
        size=cfg.training.resolution,
        repeats=getattr(cfg.data, "repeats", 1),
        center_crop=cfg.data.center_crop,
        random_flip=cfg.data.random_flip,
        mask_min_area=getattr(cfg.data, "mask_min_area", 0.1),
        mask_max_area=getattr(cfg.data, "mask_max_area", 0.5),
        mask_dilation_pct=float(getattr(cfg.data, "mask_dilation_pct", 0.10)),
        mask_max_area_pct=float(getattr(cfg.data, "mask_max_area_pct", 0.70)),
        preload_to_ram=bool(getattr(cfg.data, "preload_to_ram", False)),  # ★ NEW
    )

    val_paths  = splits["val"]
    test_paths = splits["test"]
    masks_root = Path(masks_dir)
    
    # ── Define validation prompt_map ──────────────────────────────────────
    val_prompt_map = {}
    if val_paths:
        for p in val_paths + test_paths:
            cap_path = train_dataset._find_file(Path(captions_dir), p.stem, [".txt"])
            if cap_path:
                try:
                    val_prompt_map[p.name] = cap_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

    logger.info(
        f"  train={len(splits['train'])}  "
        f"val={len(val_paths)}  test={len(test_paths)}"
    )

    # ★ FIX 2: Pre-compute VAE latents if enabled in config (eliminates VAE bottleneck)
    if getattr(cfg.data, "precompute_latents", False):
        precompute_latents(
            train_dataset,
            vae,
            accelerator,
            weight_dtype,
            batch_size=getattr(cfg.data, "latent_cache_batch_size", 8),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        collate_fn=lambda ex: collate_fn(ex, cfg.dreambooth.with_prior_preservation),
        num_workers=cfg.data.num_workers,
        pin_memory=(cfg.data.num_workers > 0),  # Only pin when using workers
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),  # Only persist workers when > 0
        # ★ FIX 4: Use config prefetch_factor instead of hardcoded 2
        prefetch_factor=(
            int(getattr(cfg.data, "prefetch_factor", 4)) if cfg.data.num_workers > 0 else None
        ),
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    max_train_steps = (
        cfg.training.max_train_steps
        or cfg.training.num_train_epochs * num_update_steps_per_epoch
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=getattr(cfg.training, "lr_num_cycles", 1),
        power=getattr(cfg.training, "lr_power", 1.0),
    )

    if train_te:
        unet, text_encoder_one, text_encoder_two, optimizer, train_loader, lr_scheduler = (
            accelerator.prepare(
                unet, text_encoder_one, text_encoder_two, optimizer, train_loader, lr_scheduler
            )
        )
    else:
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )

    # ★ FREE CPU copies after DeepSpeed moved everything to GPU
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[Memory] Freed CPU model copies after accelerator.prepare()")

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    if not cfg.training.max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = (cfg.training.resolution, cfg.training.resolution)
        ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([ids], device=accelerator.device, dtype=weight_dtype)

    # ── Pre-calculate empty prompt IDs for CFG Dropout ────────────────────
    def get_empty_ids(tokenizer):
        return tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids[0].to(accelerator.device)

    empty_ids_one = get_empty_ids(tokenizer_one)
    empty_ids_two = get_empty_ids(tokenizer_two)

    # text_encoders kept for encode_prompt calls in the training loop
    # (tokenization is now done inside the Dataset, not here)

    if accelerator.is_main_process and report_to:
        accelerator.init_trackers(
            cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info("***** Starting DreamBooth LoRA Inpainting training *****")
    logger.info(f"  Dataset dir   = {dataset_dir}")
    logger.info(f"  Instances     = {len(train_dataset)}  (train split × repeats)")
    logger.info(f"  Epochs        = {num_train_epochs}")
    logger.info(f"  Batch size    = {cfg.training.train_batch_size}")
    logger.info(f"  Grad. accum   = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total steps   = {max_train_steps}")
    
    # ⚠️ RECOMMENDED CONFIG FOR 2900 IMAGES:
    #   - num_train_epochs: 7    (currently {num_train_epochs}, suggest reduce to avoid overfitting)
    #   - validation_steps: 250   (currently in cfg, suggest reduce from 500 to catch overfitting early)
    #   - checkpointing_steps: 250 (currently in cfg, suggest reduce from 500)
    #   - dropout: 0.08           (currently 0.05, consider increase for regularization)
    if num_train_epochs > 7:
        logger.warning(
            f"⚠️  High epochs ({num_train_epochs}) for dataset size. "
            f"With 2900 images, recommend num_train_epochs ≤ 7 to prevent overfitting. "
            f"Monitor validation metrics closely."
        )
    logger.info(f"  LoRA rank     = {rank}  |  alpha = {alpha}  |  DoRA = {use_dora}")
    logger.info(f"  EDM-style     = {do_edm}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if getattr(cfg.training, "resume_from_checkpoint", None):
        ckpt = cfg.training.resume_from_checkpoint
        if ckpt == "latest":
            dirs = sorted(
                [d for d in Path(cfg.training.output_dir).iterdir()
                 if d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            ckpt = str(dirs[-1]) if dirs else None

        if ckpt:
            accelerator.load_state(ckpt)
            state_file = Path(ckpt) / "training_state.json"
            if state_file.exists():
                saved       = json.loads(state_file.read_text())
                global_step = saved["global_step"]
                first_epoch = saved["epoch"]
            else:
                global_step = int(Path(ckpt).name.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step - first_epoch * num_update_steps_per_epoch
            logger.info(f"Resumed from {ckpt}  (global_step={global_step})")

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas      = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_ts = noise_scheduler.timesteps.to(accelerator.device)
        timesteps   = timesteps.to(accelerator.device)
        step_indices = [(schedule_ts == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    snr_gamma    = getattr(cfg.training, "snr_gamma", None)
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    # ── Training loop ─────────────────────────────────────────────────────
    last_aux_log = {}  # ★ Cache aux_log between compute steps
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        early_stop = False  # ★ Reset per-epoch (prevents NameError if no val step runs)
        if train_te:
            text_encoder_one.train()
            text_encoder_two.train()
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        batches_to_skip = (
            resume_step * cfg.training.gradient_accumulation_steps
            if epoch == first_epoch else 0
        )
        if batches_to_skip > 0:
            logger.info(f"  Epoch {epoch}: skipping {batches_to_skip} batches …")
            if hasattr(accelerator, "skip_first_batches"):
                active_loader = accelerator.skip_first_batches(train_loader, batches_to_skip)
            else:
                active_loader = iter(train_loader)
                for _ in range(batches_to_skip):
                    next(active_loader, None)
        else:
            active_loader = train_loader

        for step, batch in enumerate(active_loader):
            with accelerator.accumulate(unet):
                # ── Encode images → latents (VAE is frozen) ───────────────
                # ★ FIX 3: Use cached latents if available, otherwise encode real-time
                with torch.no_grad():
                    if batch.get("latent_cached", False):
                        # Use pre-computed latents from cache
                        stems = batch["stem"]
                        latents_list = []
                        masked_latents_list = []
                        for stem in stems:
                            lat, mlat = train_dataset._latent_cache[stem]
                            latents_list.append(lat)
                            masked_latents_list.append(mlat)
                        latents = torch.stack(latents_list).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                        masked_latents = torch.stack(masked_latents_list).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                    else:
                        # Fall back to real-time VAE encoding
                        latents = vae.encode(
                            batch["pixel_values"].to(vae.device, dtype=vae.dtype, non_blocking=True)
                        ).latent_dist.sample() * vae.config.scaling_factor

                        masked_latents = vae.encode(
                            batch["masked_images"].to(vae.device, dtype=vae.dtype, non_blocking=True)
                        ).latent_dist.sample() * vae.config.scaling_factor

                mask = F.interpolate(
                    batch["masks"].to(dtype=weight_dtype, non_blocking=True),
                    size=latents.shape[-2:],
                    mode="nearest",
                )

                latents        = latents.to(weight_dtype)
                masked_latents = masked_latents.to(weight_dtype)

                # ── Noise & timesteps ─────────────────────────────────────
                noise = torch.randn_like(latents)
                bsz   = latents.shape[0]

                if not do_edm:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device,
                    ).long()
                else:
                    indices   = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if do_edm:
                    sigmas            = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)
                else:
                    inp_noisy_latents = noisy_latents

                # ── SDXL micro-conditioning ───────────────────────────────
                add_time_ids = torch.cat([
                    compute_time_ids(s, c)
                    for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ])

                # ── Text embeddings (pre-tokenized in Dataset.__getitem__) ──
                # ★ REFINED: Per-sample CFG Dropout
                ids1 = batch["input_ids_one"].to(text_encoder_one.device, non_blocking=True)
                ids2 = batch["input_ids_two"].to(text_encoder_two.device, non_blocking=True)
                
                bsz = ids1.shape[0]
                cfg_drop_rate = getattr(cfg.training, "cfg_dropout_rate", 0.0)
                dropout_mask = torch.rand(bsz, device=ids1.device) < cfg_drop_rate
                if dropout_mask.any():
                    # Replace specific samples with empty_ids
                    mask_expanded = dropout_mask.unsqueeze(-1)
                    ids1 = torch.where(mask_expanded, empty_ids_one.unsqueeze(0), ids1)
                    ids2 = torch.where(mask_expanded, empty_ids_two.unsqueeze(0), ids2)

                if not train_te:
                    with torch.no_grad():
                        cur_embeds, cur_pooled = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=None, prompt=None,
                            text_input_ids_list=[ids1, ids2],
                        )
                else:
                    cur_embeds, cur_pooled = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None, prompt=None,
                        text_input_ids_list=[ids1, ids2],
                    )

                # ── 9-channel UNet input ──────────────────────────────────
                unet_input = torch.cat([inp_noisy_latents, mask, masked_latents], dim=1)

                # ── ControlNet (Frozen extractor — no grad needed) ────────
                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        inp_noisy_latents,
                        timesteps,
                        encoder_hidden_states=cur_embeds,
                        controlnet_cond=batch["depth_values"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True),
                        added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": cur_pooled},
                        return_dict=False,
                    )
                    # Detach to ensure no graph is built back through ControlNet
                    down_block_res_samples = [s.detach() for s in down_block_res_samples]
                    mid_block_res_sample   = mid_block_res_sample.detach()

                model_pred = unet(
                    unet_input, timesteps, cur_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": cur_pooled},
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # ── EDM postconditioning ──────────────────────────────────
                weighting = None
                if do_edm:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        model_pred = model_pred * (-sigmas) + noisy_latents
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        model_pred = model_pred * (
                            -sigmas / (sigmas ** 2 + 1) ** 0.5
                        ) + (noisy_latents / (sigmas ** 2 + 1))
                    weighting = (sigmas ** -2.0).float()

                # ── Target ────────────────────────────────────────────────
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = latents if do_edm else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        latents if do_edm
                        else noise_scheduler.get_velocity(latents, noise, timesteps)
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction_type: {noise_scheduler.config.prediction_type}"
                    )

                # ── Prior preservation split ──────────────────────────────
                if cfg.dreambooth.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target,     target_prior     = torch.chunk(target,     2, dim=0)

                    if weighting is not None:
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2)
                            .reshape(target_prior.shape[0], -1), 1
                        ).mean()
                    else:
                        prior_loss = F.mse_loss(
                            model_pred_prior.float(), target_prior.float(), reduction="mean"
                        )

                # ── Diffusion MSE loss ────────────────────────────────────
                if snr_gamma is None:
                    if weighting is not None:
                        diffusion_loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2)
                            .reshape(target.shape[0], -1), 1
                        ).mean()
                    else:
                        diffusion_loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                else:
                    snr         = compute_snr(noise_scheduler, timesteps)
                    # ★ FIX: Use ones_like(snr) instead of ones_like(timesteps) to preserve dtype
                    # timesteps is .long(), snr is float. ones_like(timesteps) would be long → truncates snr_gamma
                    base_weight = (
                        torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1)
                        .min(dim=1)[0] / snr
                    )
                    mse_loss_weights = (
                        base_weight + 1
                        if noise_scheduler.config.prediction_type == "v_prediction"
                        else base_weight
                    )
                    diffusion_loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    diffusion_loss = (
                        diffusion_loss.mean(dim=list(range(1, len(diffusion_loss.shape))))
                        * mse_loss_weights
                    ).mean()

                # ── ★ NEW: Auxiliary pixel-space losses ───────────────────
                aux_loss  = torch.tensor(0.0, device=latents.device)
                aux_log: dict[str, float] = {}

                if accelerator.sync_gradients and global_step % 100 == 0:
                    logger.debug(
                        f"[Step {global_step}] aux_computer={aux_computer is not None}, "
                        f"aux_loss_every_n_steps={lw.aux_loss_every_n_steps if aux_computer else 'N/A'}, "
                        f"should_compute={aux_computer is not None and (global_step % lw.aux_loss_every_n_steps == 0)}"
                    )

                if (
                    aux_computer is not None
                    and (global_step % lw.aux_loss_every_n_steps == 0)
                ):
                    try:
                        _n_inst      = model_pred.shape[0]
                        _noisy_inst  = noisy_latents[:_n_inst]
                        _ts_inst     = timesteps[:_n_inst]
                        _latents_inst = latents[:_n_inst]
                        _mask_inst   = mask[:_n_inst]

                        aux_loss, aux_log = aux_computer.compute(
                            model_pred    = model_pred,
                            noisy_latents = _noisy_inst,
                            timesteps     = _ts_inst,
                            gt_latents    = _latents_inst,
                            mask_latent   = _mask_inst,
                            vae           = vae,
                            weights       = lw,
                            do_edm        = do_edm,
                            noise_scheduler = noise_scheduler,
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            f"[Step {global_step}] Aux loss OOM — skipping this step. "
                            f"Consider increasing aux_loss_every_n_steps or aux_resolution_scale."
                        )
                        aux_loss = torch.tensor(0.0, device=latents.device)
                        aux_log  = {}
                        torch.cuda.empty_cache()
                    # ★ Update cached aux_log
                    last_aux_log = aux_log
                    if aux_log:
                        logger.debug(f"[Step {global_step}] Aux losses computed: {list(aux_log.keys())}")

                # ── Combine all losses ────────────────────────────────────
                loss = lw.diffusion_weight * diffusion_loss + aux_loss

                if cfg.dreambooth.with_prior_preservation:
                    loss = loss + cfg.dreambooth.prior_loss_weight * prior_loss

                # ── Backward ──────────────────────────────────────────────
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    all_params = unet_params
                    if train_te:
                        all_params = itertools.chain(
                            all_params,
                            filter(lambda p: p.requires_grad, text_encoder_one.parameters()),
                            filter(lambda p: p.requires_grad, text_encoder_two.parameters()),
                        )
                    accelerator.clip_grad_norm_(all_params, cfg.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Sync & logging ────────────────────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    loss_val = loss.detach().item()
                    diff_val = diffusion_loss.detach().item()
                    lr_val   = lr_scheduler.get_last_lr()[0]

                    if global_step % cfg.logging.log_every_n_steps == 0:
                        log_dict = {
                            "train/loss":           loss_val,
                            "train/loss_diffusion": diff_val,
                            "train/lr":             lr_val,
                        }
                        # ★ Add cached auxiliary loss components (from last compute)
                        log_dict.update(last_aux_log)

                        accelerator.log(log_dict, step=global_step)

                        if use_wandb and wandb.run is not None:
                            wandb.log(
                                {**log_dict, "train/epoch": epoch, "train/step": global_step},
                                step=global_step,
                            )

                # ── Checkpoint (ALL ranks must participate for DeepSpeed) ──
                if global_step % cfg.training.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(
                        cfg.training.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(ckpt_dir)

                    # Chỉ rank 0 làm cleanup + logging
                    if accelerator.is_main_process:
                        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as fh:
                            json.dump({"global_step": global_step, "epoch": epoch}, fh)

                        ckpts = sorted(
                            [d for d in Path(cfg.training.output_dir).iterdir()
                             if d.name.startswith("checkpoint-")],
                            key=lambda d: int(d.name.split("-")[1]),
                        )
                        keep = getattr(cfg.training, "checkpoints_total_limit", 3) or 3
                        for old in ckpts[:-keep]:
                            shutil.rmtree(old)
                        logger.info(f"Saved checkpoint: {ckpt_dir}")

                if global_step % cfg.validation.validation_steps == 0:
                    accelerator.wait_for_everyone()          # sync TRƯỚC

                    # ★ FIX: Initialize early_stop for ALL ranks BEFORE the rank-specific block
                    early_stop = False

                    if accelerator.is_main_process:          # chỉ rank 0 chạy
                        torch.cuda.empty_cache()
                        gc.collect()

                        pipeline = StableDiffusionXLControlNetInpaintPipeline(
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            tokenizer=tokenizer_one,
                            tokenizer_2=tokenizer_two,
                            unet=accelerator.unwrap_model(unet),
                            controlnet=accelerator.unwrap_model(controlnet),
                            scheduler=noise_scheduler,
                        )
                        pipeline.set_progress_bar_config(disable=True)
                        val_images, val_metrics = log_validation(
                            pipeline, cfg, accelerator, epoch, global_step, weight_dtype,
                            val_paths=val_paths,
                            val_masks_root=masks_root,
                            val_depths_root=Path(depths_dir),
                            val_prompt_map=val_prompt_map,
                        )
                        del pipeline
                        torch.cuda.empty_cache()
                        
                        # ★ DISABLED: Early stopping logic (only on main process)
                        # global _BEST_VAL_LPIPS, _PATIENCE_COUNTER
                        # 
                        # max_patience = getattr(cfg.training, "early_stopping_patience", 3)
                        # if val_metrics and "val/lpips" in val_metrics:
                        #     current_lpips = val_metrics["val/lpips"]
                        #     if current_lpips < _BEST_VAL_LPIPS:
                        #         _BEST_VAL_LPIPS = current_lpips
                        #         _PATIENCE_COUNTER = 0
                        #         logger.info(f"✅ Val LPIPS improved: {current_lpips:.4f}")
                        #     else:
                        #         _PATIENCE_COUNTER += 1
                        #         logger.warning(
                        #             f"⚠️  Val LPIPS no improvement ({current_lpips:.4f} vs {_BEST_VAL_LPIPS:.4f}). "
                        #             f"Patience: {_PATIENCE_COUNTER}/{max_patience}"
                        #         )
                        #         if _PATIENCE_COUNTER >= max_patience:
                        #             logger.info(
                        #                 f"🛑 Early stopping triggered after {_PATIENCE_COUNTER} "
                        #                 f"validation runs without improvement"
                        #             )
                        #             early_stop = True

                    accelerator.wait_for_everyone()          # sync SAU
                    
                    # ★ DISABLED: Broadcast early stop decision from rank 0 to all ranks
                    # if accelerator.num_processes > 1:
                    #     early_stop_tensor = torch.tensor(
                    #         [1.0 if early_stop else 0.0],
                    #         device=accelerator.device, dtype=torch.float32,
                    #     )
                    #     torch.distributed.broadcast(early_stop_tensor, src=0)
                    #     early_stop = bool(early_stop_tensor.item() > 0.5)
                    # 
                    # if early_stop:
                    #     logger.info("🛑 Early stopping across all ranks")
                    #     break

            progress_bar.set_postfix(
                loss=loss.detach().item(),
                diff=diffusion_loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
            )
            if global_step >= max_train_steps:
                break

        # ★ FIX: Also break the outer epoch loop on early stop
        if early_stop:
            break

    # ── Save final LoRA weights ───────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped   = accelerator.unwrap_model(unet).to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_unwrapped))

        te1_lora_layers = te2_lora_layers = None
        if train_te:
            te1 = accelerator.unwrap_model(text_encoder_one).to(torch.float32)
            te2 = accelerator.unwrap_model(text_encoder_two).to(torch.float32)
            te1_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(te1))
            te2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(te2))

        # ★ FIX: Save LoRA weights using safetensors directly (same as hook fix)
        # Normalize mixed PEFT/diffusers keys
        def _normalize_lora_keys(sd):
            if sd is None:
                return sd
            return {
                k.replace(".lora_A.weight", ".lora.down.weight")
                 .replace(".lora_B.weight", ".lora.up.weight"): v
                for k, v in sd.items()
            }
        unet_lora_layers = _normalize_lora_keys(unet_lora_layers)
        te1_lora_layers  = _normalize_lora_keys(te1_lora_layers)
        te2_lora_layers  = _normalize_lora_keys(te2_lora_layers)

        all_lora_weights = {}
        if unet_lora_layers:
            for k, v in unet_lora_layers.items():
                all_lora_weights[f"unet.{k}"] = v
        if te1_lora_layers:
            for k, v in te1_lora_layers.items():
                all_lora_weights[f"text_encoder.{k}"] = v
        if te2_lora_layers:
            for k, v in te2_lora_layers.items():
                all_lora_weights[f"text_encoder_2.{k}"] = v

        final_save_path = os.path.join(cfg.training.output_dir, "pytorch_lora_weights.safetensors")
        save_file(all_lora_weights, final_save_path)
        logger.info(f"✅ Final LoRA weights saved ({len(all_lora_weights)} tensors → {final_save_path})")

        if getattr(cfg.training, "output_kohya_format", False):
            lora_sd  = load_file(f"{cfg.training.output_dir}/pytorch_lora_weights.safetensors")
            peft_sd  = convert_all_state_dict_to_peft(lora_sd)
            kohya_sd = convert_state_dict_to_kohya(peft_sd)
            save_file(kohya_sd, f"{cfg.training.output_dir}/pytorch_lora_weights_kohya.safetensors")
            logger.info("Saved Kohya-format LoRA weights.")

        vae_final = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if not getattr(cfg.model, "pretrained_vae_model_name_or_path", None) else None,
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype,
        )
        # Clear VRAM before loading validation pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pipeline = StableDiffusionXLControlNetInpaintPipeline(
            vae=vae_final,
            text_encoder=accelerator.unwrap_model(text_encoder_one),
            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=accelerator.unwrap_model(unet),
            controlnet=accelerator.unwrap_model(controlnet),
            scheduler=noise_scheduler,
        )
        pipeline.load_lora_weights(cfg.training.output_dir)
        log_validation(
            pipeline, cfg, accelerator, num_train_epochs, global_step, weight_dtype,
            val_paths=val_paths,
            val_masks_root=masks_root,
            val_depths_root=Path(depths_dir),
            val_prompt_map=val_prompt_map,
        )
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_model_card(
            cfg.training.output_dir,
            base_model=pretrained,
            dataset_dir=cfg.data.dataset_dir,
            lora_enabled=True,
            use_dora=use_dora,
            instance_prompt=getattr(cfg.dreambooth, "instance_prompt", None),
        )

        if getattr(cfg, "push_to_hub", None) and cfg.push_to_hub.enabled:
            from huggingface_hub import HfApi
            api = HfApi(token=cfg.push_to_hub.hub_token)
            api.upload_folder(
                folder_path=cfg.training.output_dir,
                repo_id=cfg.push_to_hub.hub_model_id,
                repo_type="model",
                commit_message="End of DreamBooth LoRA Inpainting training",
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
            )

        if use_wandb and wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished.")

    accelerator.end_training()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamBooth LoRA for SDXL Inpainting")
    parser.add_argument("--config", type=str, default="configs/train_config.yml",
                        help="Path to training config YAML")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID(s) to use (single-GPU mode only). Default: '0'")
    args = parser.parse_args()
    
    # ⚠️  Only set CUDA_VISIBLE_DEVICES for single-GPU mode (running python directly)
    #     When using 'accelerate launch', RANK env var is set → DON'T override device placement
    if 'RANK' not in os.environ:
        # Single-GPU mode: user runs python directly
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    cfg = load_config(args.config)
    main(cfg)