# SDXL Inpainting DreamBooth LoRA

This repository provides an optimized pipeline for fine-tuning **Stable Diffusion XL (SDXL) Inpainting** using **DreamBooth** and **LoRA**. It is designed for high-performance training (tested on multi-GPU setups) and includes specialized inference scripts for high-quality architectural and interior design inpainting.

---

## Features

- **DreamBooth LoRA Training**: Fine-tune the SDXL inpainting model with your custom dataset.
- **Advanced Control**: Integrates depth-based ControlNet conditioning for better structure preservation.
- **Optimized Performance**: Pre-configured for Multi-GPU training via `accelerate` and `DeepSpeed ZeRO-2`.
- **Premium Inference**: Specialized mask regularization and high-resolution support for realistic results.

---

## Repository Structure

- `src/train/train_dreambooth_lora.py`: Core training logic.
- `configs/train_config.yml`: Main configuration for training (paths, hyperparameters, LoRA settings).
- `src/inference/test.py`: Premium inference script with specialized post-processing.
- `train.sh`: Convenient launch script for distributed training.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd SDXL
   ```

2. **Install dependencies:**
   We recommend using a `conda` or `venv` environment.
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

Organize your dataset exactly as shown below. For every sample, ensure the files in each subdirectory share the **identical filename**.

### Structure
```text
new_data/
├── images/          # Original RGB images (e.g., room_01.jpg)
├── masks/           # Binary inpaint masks (e.g., room_01.png)
├── depth_map/       # Corresponding depth maps (e.g., room_01.png)
└── captions/        # Text files containing prompts (e.g., room_01.txt)
```

### Naming & Format Requirements
- **Synchronized Naming**: If an image is named `data_001.jpg`, then the mask must be `data_001.png`, the depth map `data_001.png`, and the caption `data_001.txt`.
- **Captions**: Each `.txt` file should contain a single plain-text string describing the image content.
- **Masks**: Grayscale images where **white (255)** represents the area to be inpainted and **black (0)** represents the area to be preserved.

---

## Training

### 1. Configuration
Review `configs/train_config.yml` to set your paths and tuning preferences. The default `dataset_dir` is set to `new_data`.

### 2. Run Training
For single GPU setups:
```bash
python src/train/train_dreambooth_lora.py --config configs/train_config.yml
```

For Multi-GPU environments (optimized for DeepSpeed):
```bash
bash train.sh
```

---

## Inference / Testing

After training, you can evaluate your LoRA weights using the specialized inference script:

1. Open `src/inference/test.py`.
2. Update the following variables:
   - `img_path`: Path to your source image.
   - `mask_path`: Path to your mask image.
   - `pipe.load_lora_weights(...)`: Path to your trained `.safetensors` file.
3. Run the script:
   ```bash
   python src/inference/test.py
   ```

The script will produce high-quality outputs including comparison previews to visualize the inpainting quality.

---

*Developed for high-fidelity interior design and architectural visualization.*
