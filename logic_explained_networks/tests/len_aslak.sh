#!/bin/bash

#SBATCH --nodes=1			                  # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=LEN                  # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --output=../slurm_out/len_%j_%a.log # Stdout and stderr file


source activate explainn

echo "LEN test"

PY_SCRIPT=../len_aslak.py

python ${PY_SCRIPT}