#!/bin/bash

#SBATCH --job-name=download_data
#SBATCH --account=aims
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB

#SBATCH --output=download_data_%j.out
#SBATCH --error=download_data_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate cellpainting

module load gcc

python download.py