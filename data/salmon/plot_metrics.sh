#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=train_explainn           # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_train_test_interpret/plot_AUCPR%j.log # Stdout and stderr file


source activate explainn

OUT_DIR="$SCRATCH/AS-TAC/ExplaiNN/single_train/${1}"
PY_SCRIPT=../../scripts/plot_metrics.py
${PY_SCRIPT} -d $OUT_DIR -o ""
