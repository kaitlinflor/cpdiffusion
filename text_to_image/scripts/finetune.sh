#!/bin/bash

#SBATCH --job-name=finetune_chem
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=4
#SBATCH --time=3-00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=finetune_chem_%j.out
#SBATCH --error=finetune_chem_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate sd

module load gcc

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/gscratch/aims/kflores3/cellpainting/datasets/preprocessed_data/3_channels"

accelerate launch --mixed_precision="fp16" --num_cpu_threads_per_process=8 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=6000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="cell-painting-chems" \
  --logging_dir="cell-painting-chems"