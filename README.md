# CP-Diffusion: Chemically Conditioned Latent Diffusion for Cell Painting

## Overview


## Environment Set Up

To replicate the environment used for this project, please run the following commands : 

```bash
git clone https://github.com/your-username/your-repo-name.git
cd cpdiffusion
conda env create -f environment.yml
```
To train diffusion model with cloome use script : 
    diffusers/examples/text_to_image/train_text_to_image_cloome.py

## Training SD
To train SD, please first export these following variables : 

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="Training dataset path"
export CHECKPOINT_PATH="Desired path to save checkpoint"
```

Enter the directory text_to_image
```bash
cd text_to_image
```

```bash
accelerate launch --mixed_precision="fp16" --num_cpu_threads_per_process=8 train_text_to_image_cloome.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resume_from_checkpoint=$CHECKPOINT_PATH \
  --use_ema \
  --caption_column text \
  --resolution=96 --center_crop --random_flip\
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=24000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="cloome-bbbc" \
  --logging_dir="log" 
```

This repository is based off of https://github.com/huggingface/diffusers/tree/main/examples/text_to_image. 