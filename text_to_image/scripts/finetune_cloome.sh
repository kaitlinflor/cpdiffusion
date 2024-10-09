#!/bin/bash

#SBATCH --job-name=train_cloome
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=slurm_output/cloome_train_%j.out
#SBATCH --error=slurm_output/cloome_train_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate sd

module load gcc

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/gscratch/aims/kflores3/cellpainting/preprocessed_data/bbbc021/train/combined"
export CHECKPOINT_PATH="/gscratch/aims/kflores3/cellpainting/diffusers/examples/text_to_image/cloome-bbbc/checkpoint-12000"

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