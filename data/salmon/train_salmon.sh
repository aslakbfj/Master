#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_salmon          # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_explainn/explainn_salmon%j_%a.log # Stdout and stderr file


source activate explainn

echo "Train (same parameters as in the preprint; it can take a few hours) and test"
PY_SCRIPT=../scripts/train.py
OUT_DIR=$SCRATCH/AS-TAC/ExplaiNN/

${PY_SCRIPT} -o ${OUT_DIR} --input-length 1000 --criterion bcewithlogits \
--patience 5 \
--num-epochs 10 \
--num-units 300 AtlanticSalmon_bins.train.tsv \
AtlanticSalmon_bins.validation.tsv

