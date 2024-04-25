#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=train_explainn           # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_actual_optimize/job%j.log # Stdout and stderr file


source activate explainn

# num-units as argument!
OPTIMIZE_SCRIPT=../../scripts/optimize_units.py
OUT_DIR="$SCRATCH/AS-TAC/ExplaiNN/optimize_units/${SLURM_JOB_ID}"
H5_FILE="$SCRATCH/AS-TAC/AS-TAC_1000bp.h5"

echo "Train (same parameters as in the preprint; it can take a few hours) and test"

${TRAIN_SCRIPT} -o ${OUT_DIR} --input-length 1000 --criterion bcewithlogits \
--patience 10 \
--num-epochs 50 \
--batch-size 200 \
--num-units ${1} AS-TAC_1000bp.train.tsv \
AS-TAC_1000bp.validation.tsv

