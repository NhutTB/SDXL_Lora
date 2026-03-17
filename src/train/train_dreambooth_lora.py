#!/usr/bin/env python
# coding=utf-8
"""
DreamBooth LoRA fine-tuning for Stable Diffusion XL Inpainting.

Dataset structure expected:
  dataset_dir/
    images/          ← original images  (e.g. 10_00339.jpg)
    masks/           ← binary masks, same filenames (e.g. 10_00339.jpg)
    prompts.json     ← list of {filename, prompt, ...}

Usage
-----
# Single GPU
python train_dreambooth_lora_sdxl_inpaint.py --config configs/train_config.yaml

# Multi-GPU
accelerate launch --num_processes=4 train_dreambooth_lora_sdxl_inpaint.py \
    --config configs/train_config.yaml
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
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub.utils import insecure_hashlib
from omegaconf import OmegaConf
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
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
    DDPMScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
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
# JSON prompt loader
# ──────────────────────────────────────────────────────────────────────────────

def load_prompt_map(json_path: str) -> dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        result = {}
        for item in data:
            fname = item.get("filename") or item.get("image")
            prompt = item.get("prompt") or item.get("caption", "")
            if fname and prompt:
                result[fname] = prompt.strip()
        return result
    elif isinstance(data, dict):
        return {k: v.strip() for k, v in data.items() if v}
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Train / Val / Test split  (80 / 10 / 10)
# ──────────────────────────────────────────────────────────────────────────────

def make_splits(
    images_dir: str,
    masks_dir: str,
    prompt_map: dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float   = 0.1,
    seed: int = 42,
) -> dict[str, list[Path]]:
    images_root = Path(images_dir)
    masks_root  = Path(masks_dir)
    _exts       = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    valid: list[Path] = []
    skipped_mask    = 0
    skipped_prompt  = 0

    for p in sorted(images_root.iterdir()):
        if p.suffix.lower() not in _exts:
            continue
        has_mask = any(
            (masks_root / (p.stem + ext)).exists()
            for ext in [p.suffix, ".png", ".jpg", ".jpeg", ".webp"]
        )
        if not has_mask:
            skipped_mask += 1
            continue
        if p.name not in prompt_map:
            skipped_prompt += 1
        valid.append(p)

    if not valid:
        raise ValueError(f"No valid (image + mask) pairs found in {images_dir}")

    if skipped_mask:
        logger.warning(f"make_splits: {skipped_mask} images skipped (missing mask)")
    if skipped_prompt:
        logger.warning(f"make_splits: {skipped_prompt} images have no JSON prompt (will use fallback)")

    rng = random.Random(seed)
    shuffled = valid.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

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
                m = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
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
        inv_sf = 1.0 / vae.config.scaling_factor
        pred_px = self._denorm(
            vae.decode(pred_x0.to(vae.dtype) * inv_sf).sample
        ).clamp(0, 1).float()

        with torch.no_grad():
            gt_px = self._denorm(
                vae.decode(gt_latents.to(vae.dtype) * inv_sf).sample
            ).clamp(0, 1).float()

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
# DreamBooth Inpainting Dataset  (JSON-based prompts)
# ──────────────────────────────────────────────────────────────────────────────

class DreamBoothInpaintingDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        prompt_map: dict[str, str],
        file_list: list[Path],
        fallback_prompt: str = "",
        class_prompt: str | None = None,
        class_data_dir: str | None = None,
        class_num: int | None = None,
        size: int = 1024,
        repeats: int = 1,
        center_crop: bool = False,
        random_flip: bool = False,
        mask_min_area: float = 0.1,
        mask_max_area: float = 0.5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.fallback_prompt = fallback_prompt
        self.class_prompt = class_prompt
        self.mask_min_area = mask_min_area
        self.mask_max_area = mask_max_area
        self.prompt_map = prompt_map

        masks_root = Path(masks_dir)
        if not masks_root.exists():
            raise ValueError(f"masks_dir does not exist: {masks_dir}")
        if not file_list:
            raise ValueError("file_list is empty — nothing to load!")

        self.instance_paths: list[Path] = list(
            itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in file_list)
        )
        self.masks_root = masks_root

        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS)
        self.train_crop   = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip   = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self._preprocess_instances()

        self.num_instance_images = len(self.instance_paths)
        self._length = self.num_instance_images

        self.class_data_root = None
        _exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if class_data_dir is not None:
            self.class_data_root = Path(class_data_dir)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            class_paths = sorted(p for p in self.class_data_root.iterdir() if p.suffix.lower() in _exts)
            self.num_class_images = min(len(class_paths), class_num) if class_num else len(class_paths)
            self.class_paths = class_paths[: self.num_class_images]
            self._length = max(self.num_class_images, self.num_instance_images)

        self.class_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    @staticmethod
    def _find_mask(masks_root: Path, img_path: Path) -> Path | None:
        for ext in [img_path.suffix, ".png", ".jpg", ".jpeg", ".webp"]:
            candidate = masks_root / (img_path.stem + ext)
            if candidate.exists():
                return candidate
        return None

    def _preprocess_instances(self):
        self.pixel_values: list[torch.Tensor]        = []
        self.mask_values: list[torch.Tensor]          = []
        self.masked_image_values: list[torch.Tensor] = []
        self.original_sizes: list[tuple[int, int]]   = []
        self.crop_top_lefts: list[tuple[int, int]]   = []
        self.prompts: list[str]                       = []

        for img_path in tqdm(self.instance_paths, desc="Loading dataset", unit="img", dynamic_ncols=True):
            prompt = self.prompt_map.get(img_path.name, self.fallback_prompt)
            self.prompts.append(prompt)

            image = exif_transpose(Image.open(img_path))
            if image.mode != "RGB":
                image = image.convert("RGB")

            self.original_sizes.append((image.height, image.width))
            image = self.train_resize(image)

            if self.random_flip and random.random() < 0.5:
                image = self.train_flip(image)

            if self.center_crop:
                y1 = max(0, int(round((image.height - self.size) / 2.0)))
                x1 = max(0, int(round((image.width - self.size) / 2.0)))
                image = self.train_crop(image)
            else:
                y1, x1, h, w = self.train_crop.get_params(image, (self.size, self.size))
                image = crop(image, y1, x1, h, w)

            self.crop_top_lefts.append((y1, x1))

            pv = self.to_tensor_norm(image)
            self.pixel_values.append(pv)

            mask_path = self._find_mask(self.masks_root, img_path)
            mask = self._load_mask(mask_path)
            self.mask_values.append(mask)
            self.masked_image_values.append(pv * (1 - mask))

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        size = self.size
        m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
        m_arr = np.array(m, dtype=np.float32) / 255.0
        m_arr = (m_arr > 0.5).astype(np.float32)
        if m_arr.max() == 0:
            return self._random_box_mask(size)
        return torch.from_numpy(m_arr).unsqueeze(0)

    def _random_box_mask(self, size: int) -> torch.Tensor:
        area     = size * size
        mask_area = random.uniform(self.mask_min_area, self.mask_max_area) * area
        h = int(random.uniform(0.2, 0.8) * size)
        w = min(int(mask_area / max(h, 1)), size)
        y0 = random.randint(0, size - h)
        x0 = random.randint(0, size - w)
        mask = np.zeros((size, size), dtype=np.float32)
        mask[y0: y0 + h, x0: x0 + w] = 1.0
        return torch.from_numpy(mask).unsqueeze(0)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        i = index % self.num_instance_images
        example = {
            "instance_images":        self.pixel_values[i],
            "instance_masks":         self.mask_values[i],
            "instance_masked_images": self.masked_image_values[i],
            "original_size":          self.original_sizes[i],
            "crop_top_left":          self.crop_top_lefts[i],
            "instance_prompt":        self.prompts[i],
        }

        if self.class_data_root:
            j = index % self.num_class_images
            cls_img = exif_transpose(Image.open(self.class_paths[j]))
            if cls_img.mode != "RGB":
                cls_img = cls_img.convert("RGB")
            cls_pv   = self.class_transform(cls_img)
            cls_mask = self._random_box_mask(self.size)
            example["class_images"]        = cls_pv
            example["class_masks"]         = cls_mask
            example["class_masked_images"] = cls_pv * (1 - cls_mask)
            example["class_prompt"]        = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation: bool = False):
    pixel_values   = [e["instance_images"] for e in examples]
    masks          = [e["instance_masks"] for e in examples]
    masked_images  = [e["instance_masked_images"] for e in examples]
    prompts        = [e["instance_prompt"] for e in examples]
    original_sizes = [e["original_size"] for e in examples]
    crop_top_lefts = [e["crop_top_left"] for e in examples]

    if with_prior_preservation:
        pixel_values   += [e["class_images"] for e in examples]
        masks          += [e["class_masks"] for e in examples]
        masked_images  += [e["class_masked_images"] for e in examples]
        prompts        += [e["class_prompt"] for e in examples]
        original_sizes += [e["original_size"] for e in examples]
        crop_top_lefts += [e["crop_top_left"] for e in examples]

    return {
        "pixel_values":   torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
        "masks":          torch.stack(masks).float(),
        "masked_images":  torch.stack(masked_images).to(memory_format=torch.contiguous_format).float(),
        "prompts":        prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
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
        cfg.model.pretrained_model_name_or_path, torch_dtype=torch_dtype
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
# Validation
# ──────────────────────────────────────────────────────────────────────────────

_METRICS_CACHE: tuple | None = None


# ★ MODIFIED: mask dilation + 4-panel comparison
def log_validation(
    pipeline, cfg, accelerator, epoch, step, weight_dtype,
    val_paths:      list[Path] | None        = None,
    val_masks_root: Path | None              = None,
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
    val_dir = Path(cfg.training.output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

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

    for idx, img_path in enumerate(tqdm(val_paths, desc="Validation", leave=False)):
        # ── Load original ─────────────────────────────────────────────────
        pil_orig = Image.open(img_path).convert("RGB").resize((res, res), Image.LANCZOS)

        # ── Load mask ─────────────────────────────────────────────────────
        pil_mask_raw = None
        if val_masks_root is not None:
            for ext in [img_path.suffix, ".png", ".jpg", ".jpeg"]:
                mp = val_masks_root / (img_path.stem + ext)
                if mp.exists():
                    marr = np.array(
                        Image.open(mp).convert("L").resize((res, res), Image.NEAREST)
                    )
                    if marr.max() > 0:
                        pil_mask_raw = Image.fromarray(marr)
                    break

        if pil_mask_raw is None:
            marr = np.zeros((res, res), dtype=np.uint8)
            marr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
            pil_mask_raw = Image.fromarray(marr)

        # ★ NEW: Dilate mask by 10%, cap at 70% of image area ─────────────
        raw_arr    = np.array(pil_mask_raw.convert("L"))
        dilated_arr = dilate_mask_pct(
            raw_arr,
            dilation_pct=mask_dilation_pct,
            max_area_pct=mask_max_area_pct,
        )
        pil_mask_dilated = Image.fromarray(dilated_arr)

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

        # ── Inpaint with DILATED mask ─────────────────────────────────────
        with autocast_ctx:
            out = pipeline(
                prompt=prompt,
                image=pil_orig,
                mask_image=pil_mask_dilated,   # ★ use dilated mask
                height=res, width=res,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

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

    return wandb_images


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load hooks
# ──────────────────────────────────────────────────────────────────────────────

def build_save_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg):
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        unet_lora_layers = te1_lora_layers = te2_lora_layers = None

        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                te1_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif text_encoder_two is not None and isinstance(
                model, type(accelerator.unwrap_model(text_encoder_two))
            ):
                te2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            else:
                raise ValueError(f"Unexpected model type: {model.__class__}")
            weights.pop()

        StableDiffusionXLInpaintPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=te1_lora_layers,
            text_encoder_2_lora_layers=te2_lora_layers,
        )

    return save_model_hook


def build_load_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg):
    def load_model_hook(models, input_dir):
        unet_ = te1_ = te2_ = None
        while models:
            m = models.pop()
            if isinstance(m, type(accelerator.unwrap_model(unet))):
                unet_ = m
            elif isinstance(m, type(accelerator.unwrap_model(text_encoder_one))):
                te1_ = m
            elif text_encoder_two is not None and isinstance(
                m, type(accelerator.unwrap_model(text_encoder_two))
            ):
                te2_ = m
            else:
                raise ValueError(f"Unexpected model type: {m.__class__}")

        lora_sd, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_sd    = {k.replace("unet.", ""): v for k, v in lora_sd.items() if k.startswith("unet.")}
        unet_sd    = convert_unet_state_dict_to_peft(unet_sd)
        incompatible = set_peft_model_state_dict(unet_, unet_sd, adapter_name="default")
        if incompatible and getattr(incompatible, "unexpected_keys", None):
            logger.warning(f"Unexpected keys when loading LoRA: {incompatible.unexpected_keys}")

        if cfg.dreambooth.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_sd, prefix="text_encoder.", text_encoder=te1_)
            if te2_ is not None:
                _set_state_dict_into_text_encoder(lora_sd, prefix="text_encoder_2.", text_encoder=te2_)

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

    text_encoder_one = text_encoder_cls_one.from_pretrained(pretrained, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(pretrained, subfolder="text_encoder_2")

    vae_path = getattr(cfg.model, "pretrained_vae_model_name_or_path", None) or pretrained
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if not getattr(cfg.model, "pretrained_vae_model_name_or_path", None) else None,
    )
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

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

    unet_lora_attn = get_lora_config(
        rank=rank_attn, alpha=rank_attn, dropout=dropout, use_dora=use_dora,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_attn)

    _cross_modules   = ["add_k_proj", "add_q_proj", "add_v_proj", "add_out_proj"]
    _cross_available = []
    for name, _ in unet.named_modules():
        for t in _cross_modules:
            if t in name and t not in _cross_available:
                _cross_available.append(t)

    if _cross_available:
        unet_lora_xattn = get_lora_config(
            rank=rank_xattn, alpha=rank_xattn, dropout=dropout, use_dora=use_dora,
            target_modules=_cross_available,
        )
        unet.add_adapter(unet_lora_xattn)
        logger.info(f"Cross-attention LoRA added: {_cross_available}  rank={rank_xattn}")
    else:
        logger.warning("Cross-attention modules not found — skipping cross-attn LoRA.")

    unet_lora_ff = get_lora_config(
        rank=rank_ff, alpha=rank_ff, dropout=dropout + 0.05, use_dora=use_dora,
        target_modules=["ff.net.0.proj", "ff.net.2"],
    )
    try:
        unet.add_adapter(unet_lora_ff)
        logger.info(f"Feed-forward LoRA added: ff.net.0.proj + ff.net.2  rank={rank_ff}")
    except Exception as e:
        logger.warning(f"FF LoRA skipped ({e}) — continuing with attn-only LoRA.")

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"UNet LoRA params: {trainable:,} trainable / {total_p:,} total "
        f"({100 * trainable / total_p:.3f}%)"
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

    # Build a simple namespace with all weights defaulting to 0 when missing
    class _LW:
        diffusion_weight  = 1.0
        pixel_weight      = 0.0
        perceptual_weight = 0.0
        clip_weight       = 0.0
        boundary_weight   = 0.0
        depth_weight      = 0.0
        semantic_weight   = 0.0
        aux_loss_every_n_steps = 1

    lw = _LW()
    if loss_weights is not None:
        for attr in vars(lw):
            val = getattr(loss_weights, attr, None)
            if val is not None:
                setattr(lw, attr, float(val) if attr != "aux_loss_every_n_steps" else int(val))

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
    dataset_dir = Path(cfg.data.dataset_dir)
    images_dir  = str(dataset_dir / getattr(cfg.data, "images_subdir", "images"))
    masks_dir   = str(dataset_dir / getattr(cfg.data, "masks_subdir",  "masks"))
    json_file   = str(dataset_dir / getattr(cfg.data, "json_file",     "prompts.json"))

    prompt_map = load_prompt_map(json_file)
    logger.info(f"Loaded {len(prompt_map)} prompt entries from {json_file}")

    splits = make_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        prompt_map=prompt_map,
        train_ratio=getattr(cfg.data, "train_ratio", 0.8),
        val_ratio=getattr(cfg.data,   "val_ratio",   0.1),
        seed=cfg.training.seed,
    )

    train_dataset = DreamBoothInpaintingDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        prompt_map=prompt_map,
        file_list=splits["train"],
        fallback_prompt=getattr(cfg.dreambooth, "instance_prompt", ""),
        class_prompt=cfg.dreambooth.class_prompt if cfg.dreambooth.with_prior_preservation else None,
        class_data_dir=cfg.dreambooth.class_data_dir if cfg.dreambooth.with_prior_preservation else None,
        class_num=cfg.dreambooth.num_class_images,
        size=cfg.training.resolution,
        repeats=getattr(cfg.data, "repeats", 1),
        center_crop=cfg.data.center_crop,
        random_flip=cfg.data.random_flip,
        mask_min_area=getattr(cfg.data, "mask_min_area", 0.1),
        mask_max_area=getattr(cfg.data, "mask_max_area", 0.5),
    )

    val_paths  = splits["val"]
    test_paths = splits["test"]
    masks_root = Path(masks_dir)

    logger.info(
        f"  train={len(splits['train'])}  "
        f"val={len(val_paths)}  test={len(test_paths)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        collate_fn=lambda ex: collate_fn(ex, cfg.dreambooth.with_prior_preservation),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
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

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    if not cfg.training.max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = (cfg.training.resolution, cfg.training.resolution)
        ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([ids], device=accelerator.device, dtype=weight_dtype)

    if not train_te:
        tokenizers    = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                emb, pooled = encode_prompt(text_encoders, tokenizers, prompt)
            return emb.to(accelerator.device), pooled.to(accelerator.device)

        has_custom_prompts = True

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
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
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
                # ── Encode images → latents ───────────────────────────────
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["masked_images"].to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                mask = F.interpolate(
                    batch["masks"].to(dtype=weight_dtype),
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

                # ── Text embeddings ───────────────────────────────────────
                if not train_te:
                    cur_embeds, cur_pooled = compute_text_embeddings(batch["prompts"])
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, batch["prompts"])
                    tokens_two = tokenize_prompt(tokenizer_two, batch["prompts"])
                    cur_embeds, cur_pooled = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None, prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )

                # ── 9-channel UNet input ──────────────────────────────────
                unet_input = torch.cat([inp_noisy_latents, mask, masked_latents], dim=1)

                model_pred = unet(
                    unet_input, timesteps, cur_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": cur_pooled},
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
                    base_weight = (
                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1)
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

                if (
                    aux_computer is not None
                    and (global_step % lw.aux_loss_every_n_steps == 0)
                ):
                    # Instance half only (for prior-preservation the batch is split)
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
                        # ★ Add individual aux loss components
                        log_dict.update(aux_log)

                        accelerator.log(log_dict, step=global_step)

                        if use_wandb and wandb.run is not None:
                            wandb.log(
                                {**log_dict, "train/epoch": epoch, "train/step": global_step},
                                step=global_step,
                            )

                    if global_step % cfg.training.checkpointing_steps == 0:
                        ckpt_dir = os.path.join(
                            cfg.training.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(ckpt_dir)
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
                        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                            pretrained,
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            unet=accelerator.unwrap_model(unet),
                            safety_checker=None,
                            torch_dtype=weight_dtype,
                        )
                        log_validation(
                            pipeline, cfg, accelerator, epoch, global_step, weight_dtype,
                            val_paths=val_paths,
                            val_masks_root=masks_root,
                            val_prompt_map=prompt_map,
                        )
                        del pipeline
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            progress_bar.set_postfix(
                loss=loss.detach().item(),
                diff=diffusion_loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
            )
            if global_step >= max_train_steps:
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

        StableDiffusionXLInpaintPipeline.save_lora_weights(
            save_directory=cfg.training.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=te1_lora_layers,
            text_encoder_2_lora_layers=te2_lora_layers,
        )

        if getattr(cfg.training, "output_kohya_format", False):
            lora_sd  = load_file(f"{cfg.training.output_dir}/pytorch_lora_weights.safetensors")
            peft_sd  = convert_all_state_dict_to_peft(lora_sd)
            kohya_sd = convert_state_dict_to_kohya(peft_sd)
            save_file(kohya_sd, f"{cfg.training.output_dir}/pytorch_lora_weights_kohya.safetensors")
            logger.info("Saved Kohya-format LoRA weights.")

        vae_final = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if not getattr(cfg.model, "pretrained_vae_model_name_or_path", None) else None,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            pretrained, vae=vae_final, torch_dtype=weight_dtype, safety_checker=None
        )
        pipeline.load_lora_weights(cfg.training.output_dir)
        log_validation(
            pipeline, cfg, accelerator, num_train_epochs, global_step, weight_dtype,
            val_paths=val_paths,
            val_masks_root=masks_root,
            val_prompt_map=prompt_map,
        )
        del pipeline

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
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    main(cfg)