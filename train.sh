#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_train.sh — 4× RTX 5090, DeepSpeed ZeRO-2
# Usage: bash launch_train.sh
# ──────────────────────────────────────────────────────────────────────────────
source /home/diffusion/miniconda3/etc/profile.d/conda.sh
conda activate task2

set -euo pipefail
 
cd /home/diffusion/SDXL_Lora
 
# ── CUDA / NCCL ──────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export NCCL_TIMEOUT=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
export NCCL_ASYNC_ERROR_HANDLING=1

# ── CPU thread control (4 GPU × 1 thread = 4 total) ──────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_ARENA_MAX=1

# ── WandB (đặt key ở đây thay vì trong YAML) ────────────────────────────────
export WANDB_API_KEY=""

# ── Pre-launch RAM check ──────────────────────────────────────────────────────
RAM_PERCENT=$(free | awk 'NR==2{printf("%.0f", $3/$2 * 100.0)}')
if [ "$RAM_PERCENT" -gt 70 ]; then
    echo "[WARNING] RAM usage already at ${RAM_PERCENT}% — consider freeing memory before training"
fi

# ── Launch (with nice -n 5 to protect system) ────────────────────────────────
nice -n 5 accelerate launch --config_file configs/accelerate_ds_zero2.yaml \
    src/train/train_dreambooth_lora.py --config configs/train_config.yml
    2>&1 | tee train_log.txt