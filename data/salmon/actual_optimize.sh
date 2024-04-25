#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=opt_units                # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_actual_optimize/job%j.log # Stdout and stderr file


source activate explainn

# error if no argument is given
if [ -z "$1" ]
  then
    echo "No argument supplied! Either 21_25_10, 21_25_05, 10 or 05"
    exit 1
fi

OPTIMIZE_SCRIPT=../../scripts/optimize_units.py
OUT_DIR="$SCRATCH/AS-TAC/ExplaiNN/optimize_units/${1}_${SLURM_JOB_ID}"
H5_FILE="$SCRATCH/AS-TAC/AS-TAC_${1}.h5"

${OPTIMIZE_SCRIPT} ${H5_FILE} ${OUT_DIR} --input-length 1000 --criterion bcewithlogits \
--patience 1 \
--num-epochs 1 \
--batch-size 100 \
--num-units 100  \
-t \
--lr 0.003
