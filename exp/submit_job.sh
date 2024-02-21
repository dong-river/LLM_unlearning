#!/bin/bash

# Directory containing the .slurm files
DIRECTORY="/home1/r/riverd/LLM_unlearning/exp2"

# Loop through each .slurm file in the directory
for slurm_file in "$DIRECTORY"/kl*.slurm; do
    # Run batch command on the file
    sbatch "$slurm_file"
done