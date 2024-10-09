#!/bin/bash

#SBATCH --job-name=calculate_metrics
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=calc_metrics_%j.out
#SBATCH --error=calc_metrics_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate sd

module load gcc

# python calculate_metrics.py \
#     --gt_folder /gscratch/aims/kflores3/cellpainting/preprocessed_data/bbbc021/train/combined \
#     --gen_folder /gscratch/aims/kflores3/cellpainting/diffusers/examples/text_to_image/train/combined


python calculate_metrics.py \
    --gt_folder /gscratch/aims/kflores3/cellpainting/data/preprocessed_data/bbbc021_all/train/combined \
    --gen_folder /gscratch/scrubbed/kflores3/generated_data_2/cp_data/train/combined
    # --gen_folder /gscratch/aims/kflores3/cellpainting/text_to_image/train/combined