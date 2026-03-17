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
    """
    Load {filename -> prompt} mapping from the JSON file.

    Accepts both:
      - list of {"filename": "x.jpg", "prompt": "...", ...}
      - dict of {"x.jpg": "..."}
    """
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
    # test = 1 - train - val
    seed: int = 42,
) -> dict[str, list[Path]]:
    """
    Scan images_dir, keep only files that also have a matching mask, then
    shuffle deterministically and split into train / val / test.

    Returns
    -------
    {"train": [...], "val": [...], "test": [...]}
    Each list contains Path objects pointing into images_dir.
    """
    images_root = Path(images_dir)
    masks_root  = Path(masks_dir)
    _exts       = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    valid: list[Path] = []
    skipped_mask    = 0
    skipped_prompt  = 0

    for p in sorted(images_root.iterdir()):
        if p.suffix.lower() not in _exts:
            continue
        # Must have a corresponding mask
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

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = valid.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # test gets the remainder so rounding never drops samples
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
# DreamBooth Inpainting Dataset  (JSON-based prompts)
# ──────────────────────────────────────────────────────────────────────────────

class DreamBoothInpaintingDataset(Dataset):
    """
    Dataset for DreamBooth LoRA Inpainting.

    Expected layout:
        images_dir/          ← RGB images
        masks_dir/           ← binary masks, SAME filenames as images
        json_file            ← [{"filename": "x.jpg", "prompt": "..."}, ...]

    Prompt lookup: json_file filename → prompt.
    Falls back to `fallback_prompt` if a filename is not in the JSON.
    For prior-preservation: class_data_dir images use class_prompt.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        prompt_map: dict[str, str],               # pre-loaded {filename -> prompt}
        file_list: list[Path],                     # pre-computed split (from make_splits)
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

        # Repeat for DreamBooth-style oversampling
        self.instance_paths: list[Path] = list(
            itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in file_list)
        )
        self.masks_root = masks_root

        # ── Image transforms ──────────────────────────────────────────────
        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS)
        self.train_crop   = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip   = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Pre-process & cache instance images
        self._preprocess_instances()

        self.num_instance_images = len(self.instance_paths)
        self._length = self.num_instance_images

        # ── Class images (prior preservation) ────────────────────────────
        self.class_data_root = None
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

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_mask(masks_root: Path, img_path: Path) -> Path | None:
        """Return mask path for img_path, or None if not found."""
        # Try same extension first, then .png, then any image extension
        for ext in [img_path.suffix, ".png", ".jpg", ".jpeg", ".webp"]:
            candidate = masks_root / (img_path.stem + ext)
            if candidate.exists():
                return candidate
        return None

    # ── Pre-processing ────────────────────────────────────────────────────

    def _preprocess_instances(self):
        self.pixel_values: list[torch.Tensor]        = []
        self.mask_values: list[torch.Tensor]          = []
        self.masked_image_values: list[torch.Tensor] = []
        self.original_sizes: list[tuple[int, int]]   = []
        self.crop_top_lefts: list[tuple[int, int]]   = []
        self.prompts: list[str]                       = []

        for img_path in tqdm(self.instance_paths, desc="Loading dataset", unit="img", dynamic_ncols=True):
            # ── Prompt ───────────────────────────────────────────────────
            prompt = self.prompt_map.get(img_path.name, self.fallback_prompt)
            self.prompts.append(prompt)

            # ── Image ────────────────────────────────────────────────────
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

            # ── Mask ─────────────────────────────────────────────────────
            mask_path = self._find_mask(self.masks_root, img_path)
            mask = self._load_mask(mask_path)
            self.mask_values.append(mask)

            # Masked image = image * (1 - mask)
            self.masked_image_values.append(pv * (1 - mask))

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """Load mask from file → [1, H, W] binary float tensor."""
        size = self.size
        m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
        m_arr = np.array(m, dtype=np.float32) / 255.0
        m_arr = (m_arr > 0.5).astype(np.float32)
        if m_arr.max() == 0:
            # Blank mask → fall back to random box
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

    # ── Dataset protocol ──────────────────────────────────────────────────

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

    has_fp16  = torch.cuda.is_available() or torch.backends.mps.is_available()
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
    """
    Explicitly initialise a WandB run on the main process.

    Config keys used (all optional with sensible defaults):
      cfg.logging.wandb_project    (default: "sdxl-inpaint-dreambooth")
      cfg.logging.wandb_entity     (default: None  → your default entity)
      cfg.logging.wandb_run_name   (default: cfg.logging.run_name)
      cfg.logging.wandb_tags       (default: [])
      cfg.logging.wandb_notes      (default: "")
      cfg.logging.wandb_api_key    (default: None  → uses WANDB_API_KEY env var)
    """
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
    """Log a list of PIL images to WandB (safe no-op if wandb not active)."""
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
    """
    Lazy-import metrics libs. Returns (lpips_fn, ssim_fn, psnr_fn) or None for
    each that is unavailable. Install with:
        pip install lpips torchmetrics
    """
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
    """PIL → float32 tensor [1, 3, H, W] in [-1, 1] for LPIPS, [0, 1] not needed."""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _pil_to_01(pil_img: Image.Image, device) -> torch.Tensor:
    """PIL → float32 tensor [1, 3, H, W] in [0, 1] for SSIM / PSNR."""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_metrics(
    pred: Image.Image,
    gt: Image.Image,
    mask: Image.Image,
    device,
    lpips_fn=None,
    ssim_fn=None,
    psnr_fn=None,
) -> dict:
    """
    Compute LPIPS / SSIM / PSNR only on the masked (inpainted) region.

    pred  : inpainted output
    gt    : ground-truth (original image)
    mask  : binary PIL mask (255 = inpainted region)
    """
    results = {}

    # Boolean mask for the inpainted region [H, W]
    mask_np   = (np.array(mask.convert("L")) > 127)
    has_region = mask_np.any()

    if not has_region:
        return results  # blank mask — nothing to measure

    # Crop both images to the bounding box of the mask region for efficiency
    ys, xs = np.where(mask_np)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    pred_crop = pred.crop((x1, y1, x2, y2))
    gt_crop   = gt.crop((x1, y1, x2, y2))

    if lpips_fn is not None:
        try:
            lpips_fn = lpips_fn.to(device)
            with torch.no_grad():
                lp = lpips_fn(
                    _pil_to_chw(pred_crop, device),
                    _pil_to_chw(gt_crop,   device),
                ).item()
            results["val/lpips"] = lp
        except Exception as e:
            logger.warning(f"LPIPS error: {e}")

    if ssim_fn is not None:
        try:
            with torch.no_grad():
                sv = ssim_fn(
                    _pil_to_01(pred_crop, device),
                    _pil_to_01(gt_crop,   device),
                    data_range=1.0,
                ).item()
            results["val/ssim"] = sv
        except Exception as e:
            logger.warning(f"SSIM error: {e}")

    if psnr_fn is not None:
        try:
            with torch.no_grad():
                pv = psnr_fn(
                    _pil_to_01(pred_crop, device),
                    _pil_to_01(gt_crop,   device),
                    data_range=1.0,
                ).item()
            results["val/psnr"] = pv
        except Exception as e:
            logger.warning(f"PSNR error: {e}")

    return results


def make_comparison_image(original: Image.Image,
                           mask: Image.Image,
                           result: Image.Image) -> Image.Image:
    """
    Stitch 3 images side-by-side:
        [Original] | [Mask (red overlay)] | [Inpainted result]
    """
    w, h = original.size

    # Overlay mask in red on the original for clarity
    orig_arr = np.array(original.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))
    overlay  = orig_arr.copy()
    overlay[mask_arr > 127] = [255, 80, 80]   # red tint on masked region
    overlay_img = Image.fromarray(
        (orig_arr * 0.5 + overlay * 0.5).astype(np.uint8)
    )

    combined = Image.new("RGB", (w * 3, h))
    combined.paste(original,    (0,     0))
    combined.paste(overlay_img, (w,     0))
    combined.paste(result,      (w * 2, 0))
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

# Module-level cache so metrics libs are only imported once
_METRICS_CACHE: tuple | None = None


def log_validation(pipeline, cfg, accelerator, epoch, step, weight_dtype,
                   val_paths: list[Path] | None = None,
                   val_masks_root: Path | None  = None,
                   val_prompt_map: dict[str, str] | None = None):
    if not val_paths:
        logger.info("No val split — skipping validation.")
        return []

    logger.info(f"Running validation (epoch {epoch}, step {step}, {len(val_paths)} val images) …")
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

    # Load metrics libs once
    global _METRICS_CACHE
    if _METRICS_CACHE is None:
        _METRICS_CACHE = _try_import_metrics()
    lpips_fn, ssim_fn, psnr_fn = _METRICS_CACHE

    # Run on ALL val images for metrics, but only log num_validation_images to WandB
    num_to_show    = min(cfg.validation.num_validation_images, len(val_paths))
    all_metrics: list[dict] = []
    wandb_images:  list     = []
    wandb_prompts: list     = []

    for idx, img_path in enumerate(tqdm(val_paths, desc="Validation", leave=False)):
        # ── Load original image ───────────────────────────────────────────
        pil_orig = Image.open(img_path).convert("RGB").resize((res, res), Image.LANCZOS)

        # ── Load mask ─────────────────────────────────────────────────────
        pil_mask = None
        if val_masks_root is not None:
            for ext in [img_path.suffix, ".png", ".jpg", ".jpeg"]:
                mp = val_masks_root / (img_path.stem + ext)
                if mp.exists():
                    marr = np.array(Image.open(mp).convert("L").resize((res, res), Image.NEAREST))
                    if marr.max() > 0:
                        pil_mask = Image.fromarray(marr)
                    break
        if pil_mask is None:
            marr = np.zeros((res, res), dtype=np.uint8)
            marr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
            pil_mask = Image.fromarray(marr)

        # ── Prompt ────────────────────────────────────────────────────────
        prompt = (val_prompt_map or {}).get(img_path.name, "interior design, high quality")

        # ── Inpaint ───────────────────────────────────────────────────────
        with autocast_ctx:
            out = pipeline(
                prompt=prompt,
                image=pil_orig,
                mask_image=pil_mask,
                height=res, width=res,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

        out.save(val_dir / f"epoch{epoch:04d}_step{step:07d}_{img_path.stem}.png")

        # ── Metrics (on all val images) ───────────────────────────────────
        m = compute_metrics(
            pred=out, gt=pil_orig, mask=pil_mask,
            device=accelerator.device,
            lpips_fn=lpips_fn, ssim_fn=ssim_fn, psnr_fn=psnr_fn,
        )
        all_metrics.append(m)

        # ── Collect for WandB image log (first N only) ────────────────────
        if idx < num_to_show:
            comparison = make_comparison_image(pil_orig, pil_mask, out)
            comparison.save(val_dir / f"epoch{epoch:04d}_step{step:07d}_{img_path.stem}_cmp.png")
            wandb_images.append(comparison)
            wandb_prompts.append(f"[orig | mask overlay | inpaint]  {prompt}")

    # ── Aggregate metrics across all val images ───────────────────────────
    agg = {}
    for key in ["val/lpips", "val/ssim", "val/psnr"]:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            agg[key]              = float(np.mean(vals))
            agg[key + "_std"]     = float(np.std(vals))
            agg[key + "_best"]    = float(np.min(vals) if "lpips" in key else np.max(vals))

    if agg:
        logger.info(
            f"  val metrics → "
            + "  ".join(f"{k}={v:.4f}" for k, v in agg.items() if "_std" not in k and "_best" not in k)
        )

    # ── Log to WandB ──────────────────────────────────────────────────────
    if accelerator.is_main_process and is_wandb_available() and wandb.run is not None:
        log_dict = {**agg, "val/epoch": epoch}
        # Comparison images
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
    # Use "wandb" as tracker only if report_to == "wandb"; accelerate will init
    # its own wandb wrapper, but we also call init_wandb() for full control.
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

    # ── WandB explicit init ───────────────────────────────────────────────
    use_wandb = cfg.logging.report_to == "wandb"
    if use_wandb and accelerator.is_main_process:
        init_wandb(cfg, accelerator)

    # ── Generate class images if prior preservation enabled ───────────────
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

    # ── Weight dtype ──────────────────────────────────────────────────────
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # ── xFormers ─────────────────────────────────────────────────────────
    if getattr(cfg.training, "enable_xformers", False):
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers enabled.")
        else:
            logger.warning("xFormers requested but not available.")

    # ── Gradient checkpointing ────────────────────────────────────────────
    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── LoRA / DoRA on UNet ───────────────────────────────────────────────
    rank     = cfg.lora.rank
    alpha    = getattr(cfg.lora, "alpha", rank)
    dropout  = getattr(cfg.lora, "dropout", 0.0)
    use_dora = getattr(cfg.lora, "use_dora", False)

    # Rank riêng cho từng nhóm layer — tránh overfit với dataset nhỏ
    rank_attn = rank                                          # self-attention (quan trọng nhất)
    rank_ff   = getattr(cfg.lora, "rank_ff",   rank // 2)   # feed-forward   (học texture/style)
    rank_xattn= getattr(cfg.lora, "rank_xattn",rank // 2)   # cross-attention (text↔image)

    # ── Nhóm 1: Self-attention projection (rank đầy đủ) ──────────────────
    # Học "vật thể nào liên quan đến vật thể nào trong ảnh"
    unet_lora_attn = get_lora_config(
        rank=rank_attn, alpha=rank_attn, dropout=dropout, use_dora=use_dora,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_attn)

    # ── Nhóm 2: Cross-attention (rank/2) ─────────────────────────────────
    # Học "prompt ảnh hưởng vào ảnh như thế nào" — text↔image
    # add_k_proj / add_v_proj là key/value từ text encoder sang UNet
    _cross_modules = ["add_k_proj", "add_q_proj", "add_v_proj", "add_out_proj"]
    _cross_available = []
    for name, module in unet.named_modules():
        for target in _cross_modules:
            if target in name and target not in _cross_available:
                _cross_available.append(target)

    if _cross_available:
        unet_lora_xattn = get_lora_config(
            rank=rank_xattn, alpha=rank_xattn, dropout=dropout, use_dora=use_dora,
            target_modules=_cross_available,
        )
        unet.add_adapter(unet_lora_xattn)
        logger.info(f"Cross-attention LoRA added: {_cross_available}  rank={rank_xattn}")
    else:
        logger.warning("Cross-attention modules not found — skipping cross-attn LoRA.")

    # ── Nhóm 3: Feed-forward (rank/2) ────────────────────────────────────
    # Học texture, material, lighting style của interior design
    _ff_modules = []
    for name, _ in unet.named_modules():
        if "ff.net" in name:
            if "0.proj" in name and "ff_net_0_proj" not in _ff_modules:
                _ff_modules.append("proj")   # linear inside ff block
                break

    unet_lora_ff = get_lora_config(
        rank=rank_ff, alpha=rank_ff, dropout=dropout + 0.05, use_dora=use_dora,
        # dropout cao hơn 1 chút cho FF vì dễ overfit nhất
        target_modules=["ff.net.0.proj", "ff.net.2"],
    )
    try:
        unet.add_adapter(unet_lora_ff)
        logger.info(f"Feed-forward LoRA added: ff.net.0.proj + ff.net.2  rank={rank_ff}")
    except Exception as e:
        logger.warning(f"FF LoRA skipped ({e}) — continuing with attn-only LoRA.")

    # Log tổng số trainable params
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"UNet LoRA params: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.3f}%)"
    )

    # ── LoRA on text encoders (optional) ─────────────────────────────────
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

    # ── Save / Load hooks ─────────────────────────────────────────────────
    accelerator.register_save_state_pre_hook(
        build_save_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg)
    )
    accelerator.register_load_state_pre_hook(
        build_load_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg)
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    unet_params         = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize  = [{"params": unet_params, "lr": cfg.training.learning_rate}]

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

    # ── Load prompt map & build splits (80 / 10 / 10) ────────────────────
    dataset_dir  = Path(cfg.data.dataset_dir)
    images_dir   = str(dataset_dir / getattr(cfg.data, "images_subdir", "images"))
    masks_dir    = str(dataset_dir / getattr(cfg.data, "masks_subdir",  "masks"))
    json_file    = str(dataset_dir / getattr(cfg.data, "json_file",     "prompts.json"))

    prompt_map = load_prompt_map(json_file)
    logger.info(f"Loaded {len(prompt_map)} prompt entries from {json_file}")

    splits = make_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        prompt_map=prompt_map,
        train_ratio=getattr(cfg.data, "train_ratio", 0.8),
        val_ratio=getattr(cfg.data,   "val_ratio",   0.1),
        # test = remainder
        seed=cfg.training.seed,
    )

    # ── Dataset & DataLoader ──────────────────────────────────────────────
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

    # Val / test splits kept as plain path lists — used only for inference
    val_paths  = splits["val"]
    test_paths = splits["test"]
    masks_root = Path(masks_dir)

    logger.info(
        f"  train={len(splits['train'])}  "
        f"val={len(val_paths)}  "
        f"test={len(test_paths)}"
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

    # ── Prepare with Accelerator ──────────────────────────────────────────
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

    # ── Pre-compute text embeddings (frozen text encoder) ────────────────
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

        # All prompts are per-image from JSON, so always encode per-batch
        has_custom_prompts = True

    # ── Trackers (Accelerate's built-in tracker wrapper) ──────────────────
    if accelerator.is_main_process and report_to:
        accelerator.init_trackers(
            cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info("***** Starting DreamBooth LoRA Inpainting training *****")
    logger.info(f"  Dataset dir   = {dataset_dir}")
    logger.info(f"  Images        = {images_dir}")
    logger.info(f"  Masks         = {masks_dir}")
    logger.info(f"  JSON prompts  = {json_file}")
    logger.info(f"  Instances     = {len(train_dataset)}  (train split × repeats)")
    logger.info(f"  Val images    = {len(val_paths)}")
    logger.info(f"  Test images   = {len(test_paths)}")
    logger.info(f"  Epochs        = {num_train_epochs}")
    logger.info(f"  Batch size    = {cfg.training.train_batch_size}")
    logger.info(f"  Grad. accum   = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total steps   = {max_train_steps}")
    logger.info(f"  LoRA rank     = {rank}  |  alpha = {alpha}  |  DoRA = {use_dora}")
    logger.info(f"  Prior preserv = {cfg.dreambooth.with_prior_preservation}")
    logger.info(f"  Train TE      = {train_te}")
    logger.info(f"  EDM-style     = {do_edm}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    # ── Resume ────────────────────────────────────────────────────────────
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

    # ── EDM sigma helper ──────────────────────────────────────────────────
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

        batches_to_skip = resume_step * cfg.training.gradient_accumulation_steps if epoch == first_epoch else 0
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

                # ── Text embeddings (per-batch, from JSON prompts) ────────
                if not train_te:
                    cur_embeds, cur_pooled = compute_text_embeddings(batch["prompts"])
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, batch["prompts"])
                    tokens_two = tokenize_prompt(tokenizer_two, batch["prompts"])
                    cur_embeds, cur_pooled = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )

                # ── 9-channel UNet input ──────────────────────────────────
                unet_input = torch.cat([inp_noisy_latents, mask, masked_latents], dim=1)

                model_pred = unet(
                    unet_input,
                    timesteps,
                    cur_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": cur_pooled},
                    return_dict=False,
                )[0]

                # ── EDM postconditioning ──────────────────────────────────
                weighting = None
                if do_edm:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        model_pred = model_pred * (-sigmas) + noisy_latents
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        model_pred = model_pred * (-sigmas / (sigmas ** 2 + 1) ** 0.5) + (
                            noisy_latents / (sigmas ** 2 + 1)
                        )
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
                    raise ValueError(f"Unknown prediction_type: {noise_scheduler.config.prediction_type}")

                # ── Prior preservation split ──────────────────────────────
                if cfg.dreambooth.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target,     target_prior     = torch.chunk(target, 2, dim=0)

                    if weighting is not None:
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2)
                            .reshape(target_prior.shape[0], -1), 1
                        ).mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # ── Instance loss ─────────────────────────────────────────
                if snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2)
                            .reshape(target.shape[0], -1), 1
                        ).mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr         = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    mse_loss_weights = (
                        base_weight + 1 if noise_scheduler.config.prediction_type == "v_prediction"
                        else base_weight
                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights).mean()

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
                    lr_val   = lr_scheduler.get_last_lr()[0]

                    if global_step % cfg.logging.log_every_n_steps == 0:
                        # Accelerate tracker (covers wandb / tensorboard)
                        accelerator.log(
                            {"train/loss": loss_val, "train/lr": lr_val},
                            step=global_step,
                        )
                        # Direct wandb log (more granular metrics)
                        if use_wandb and wandb.run is not None:
                            wandb.log({
                                "train/loss":  loss_val,
                                "train/lr":    lr_val,
                                "train/epoch": epoch,
                                "train/step":  global_step,
                            }, step=global_step)

                    # ── Checkpoint ────────────────────────────────────────
                    if global_step % cfg.training.checkpointing_steps == 0:
                        ckpt_dir = os.path.join(cfg.training.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(ckpt_dir)
                        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as fh:
                            json.dump({"global_step": global_step, "epoch": epoch}, fh)

                        # Prune old checkpoints
                        ckpts = sorted(
                            [d for d in Path(cfg.training.output_dir).iterdir()
                             if d.name.startswith("checkpoint-")],
                            key=lambda d: int(d.name.split("-")[1]),
                        )
                        keep = getattr(cfg.training, "checkpoints_total_limit", 3) or 3
                        for old in ckpts[:-keep]:
                            shutil.rmtree(old)
                        logger.info(f"Saved checkpoint: {ckpt_dir}")

                    # ── Validation ────────────────────────────────────────
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

            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
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

        # Kohya export
        if getattr(cfg.training, "output_kohya_format", False):
            lora_sd = load_file(f"{cfg.training.output_dir}/pytorch_lora_weights.safetensors")
            peft_sd  = convert_all_state_dict_to_peft(lora_sd)
            kohya_sd = convert_state_dict_to_kohya(peft_sd)
            save_file(kohya_sd, f"{cfg.training.output_dir}/pytorch_lora_weights_kohya.safetensors")
            logger.info("Saved Kohya-format LoRA weights.")

        # Final validation
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
            validation_prompt=None,
        )

        # Push to Hub
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

        # ── Close WandB run ───────────────────────────────────────────────
        if use_wandb and wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished.")

    accelerator.end_training()
    logger.info("Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamBooth LoRA for SDXL Inpainting")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    main(cfg)