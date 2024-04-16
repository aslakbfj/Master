#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=train_explainn           # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_explainn/explainn_salmon%j.log # Stdout and stderr file


source activate explainn

echo $SCRATCH
echo "Train (same parameters as in the preprint; it can take a few hours) and test"
PY_SCRIPT=../../scripts/train.py
OUT_DIR="$SCRATCH/AS-TAC/ExplaiNN/single_train/${SLURM_ARRAY_JOB_ID}

${PY_SCRIPT} -o ${OUT_DIR} --input-length 1000 --criterion bcewithlogits \
--patience 10 \
--num-epochs 50 \
--batch-size 200 \
--num-units $1 AS-TAC_1000bp.train.tsv \
AS-TAC_1000bp.validation.tsv

echo "training done"

echo "testing..."



