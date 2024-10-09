#!/bin/bash

#SBATCH --job-name=calculate_metrics_csv
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=calc_metrics_csv_%j.out
#SBATCH --error=calc_metrics_csv_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate sd

module load gcc

# Update to use calculate_metrics_csv.py with CSV input
python calculate_metrics_csv.py \
    --gt_csv /gscratch/aims/kflores3/cellpainting/data/preprocessed_data/bbbc021/train_metadata.csv \
    --image_folder /gscratch/aims/kflores3/cellpainting/data/preprocessed_data/bbbc021/train/combined \
    --gen_folder /gscratch/aims/kflores3/cellpainting/data/generated_data/cloome_12000/train/combined

    # --gen_folder /gscratch/aims/kflores3/cellpainting/text_to_image/train/combined
    # --gen_folder /gscratch/aims/kflores3/cellpainting/data/generated_data/cloome_12000/train/combined
    # --gen_folder /gscratch/aims/kflores3/cellpainting/data/generated_data/IMPA/combined
