#!/bin/bash

#SBATCH --job-name=cellprofiler
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB

#SBATCH --output=cellprofiler_%j.out
#SBATCH --error=cellprofiler_%j.err

echo "Running script: $0"
cat $0

conda init bash
source ~/.bashrc


singularity run --bind /gscratch/aims/kflores3/cellpainting/analysis:/CellPainting https://depot.galaxyproject.org/singularity/cellprofiler:4.2.6--py39hf95cd2a_0 bash

export CELL_PAINTING_HOME=/CellPainting

# Define an array of cppipe files and their corresponding output directories
declare -A cppipe_outputs=(
    ["test_gen.cppipe"]="output_test_gen"
    ["test_gt.cppipe"]="output_test_gt"
    ["train_gen.cppipe"]="output_train_gen"
    ["train_gt.cppipe"]="output_train_gt"
)


export CELL_PAINTING_HOME=/CellPainting


declare -A cppipe_outputs=(
    ["test_gen.cppipe"]="output_test_gen"
    ["train_gen.cppipe"]="output_train_gen"
)

for cppipe_file in "${!cppipe_outputs[@]}"; do
    output_dir=${cppipe_outputs[$cppipe_file]}
    
    echo "Running CellProfiler for $cppipe_file with output to $output_dir"

    cellprofiler -c -r -i $CELL_PAINTING_HOME/ \
        -p $CELL_PAINTING_HOME/CellProfiler/$cppipe_file \
        -o $CELL_PAINTING_HOME/cp_outputs/$output_dir \
        --plugins=cellprofiler_core.worker.Worker,n_workers=4
done

# GENERATED IMAGES

# cellprofiler -c -r -i $CELL_PAINTING_HOME/ \
#     -p $CELL_PAINTING_HOME/CellProfiler/test_gen.cppipe \
#     -o $CELL_PAINTING_HOME/cp_outputs/output_generated \
#     --plugins=cellprofiler_core.worker.Worker,n_workers=4


