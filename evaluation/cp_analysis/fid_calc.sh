#!/bin/bash

#SBATCH --job-name=fid_calc
#SBATCH --account=aims
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=fid_calc_%j.out
#SBATCH --error=fid_calc_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc
conda activate sd

module load gcc

python fid_calc.py