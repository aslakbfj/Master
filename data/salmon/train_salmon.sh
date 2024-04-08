#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_salmon          # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_out/explainn_sal%j_%a.log # Stdout and stderr file


source activate explainn


#### AI-TAC dataset needed
AI_TAC=$SCRATCH/ExplaiNN/tutorial/AI-TAC
mm10=$SCRATCH/ExplaiNN/genomes/mm10



awk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.1 ? ".validation" : ".train");
print > f}' AtlanticSalmon_bins.tsv



echo "Train (same parameters as in the preprint; it can take a few hours) and test"
PY_SCRIPT=../scripts/train.py
OUT_DIR=$SCRATCH/ExplaiNN/salmon/ExplaiNN/salmon
${PY_SCRIPT} -o ${OUT_DIR} --input-length 1000 --criterion bcewithlogits \
--patience 5 \
--num-epochs 10 \
--num-units 300 AtlanticSalmon_bins.tsv.train \
AtlanticSalmon_bins.tsv.validation

