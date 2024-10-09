#!/bin/bash

#SBATCH --job-name=clustering
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=clustering_%j.out
#SBATCH --error=clustering_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc

python clustering.py