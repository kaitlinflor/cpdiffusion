#!/bin/bash

#SBATCH --job-name=gen_examples_
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=slurm_output/gen_examples_%j.out
#SBATCH --error=slurm_output/gen_examples_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate sd

module load gcc

python generate_examples_bbbc.py \
  --chemicals_path /gscratch/aims/kflores3/cellpainting/data/preprocessed_data/bbbc021_1000/test_metadata.csv \
  --save_dir /gscratch/scrubbed/kflores3/generated_data_2/cp_data/test \
  --num_images 500 \
  --model_path sample-cloome-bbbc