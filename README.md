# SDXL Inpainting DreamBooth LoRA

This repository provides scripts for fine-tuning Stable Diffusion XL (SDXL) Inpainting using DreamBooth and LoRA, along with a testing script to run inference on the trained model.

## Features
- **DreamBooth LoRA Training**: Fine-tune the SDXL inpainting model with your own dataset.
- **Flexible Configuration**: Easily configure learning rates, batch size, and dataset paths via `configs/train_config.yml`.
- **High-Resolution Inference**: Use `src/inference/test.py` to inpaint at any resolution, with specialized mask regularization to ensure perfect boundaries and realistic generation.

## Repository Structure

- `src/train/train_dreambooth_lora.py`: The main training script. Supports single GPU and distributed Multi-GPU training via `accelerate`.
- `configs/train_config.yml`: Configuration file for the training pipeline (data paths, LoRA hyperparameters, validation settings, etc.).
- `src/inference/test.py`: Inference script for generating high-quality inpainting results using the fine-tuned LoRA weights.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd SDXL
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Training

### 1. Data Preparation
Organize your data as specified in `configs/train_config.yml`. The default layout expects:
```
data/
  interior/
    images/         <- RGB images (e.g., image1.jpg)
    masks/          <- Binary masks (e.g., image1.png)
    prompts.json    <- JSON file containing prompts for each image
```

### 2. Run Training
Run the training script using Python for a single GPU:
```bash
python src/train/train_dreambooth_lora.py --config configs/train_config.yml
```

Or scale up with `accelerate` for Multi-GPU environments:
```bash
accelerate launch --num_processes=4 src/train/train_dreambooth_lora.py --config configs/train_config.yml
```

## Inference / Testing

Once the model is fine-tuned, you can test it on your images. 

1. Edit `src/inference/test.py` to correctly point to your custom image (`img_path`), your mask (`mask_path`), and the newly generated LoRA weights in the `pipe.load_lora_weights(...)` line.
2. Update the `prompt` and `negative_prompt`.
3. Run the script:
   ```bash
   python src/inference/test.py
   ```
   The script will generate `result_inpainting_*.png` and `comparison_preview_*.png`.
