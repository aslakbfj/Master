#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_interpret          # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_out/compare_%j_%a.log # Stdout and stderr file


source activate explainn


OUT_DIR=./ExplaiNN/AI-TAC

script=./compare_tomtom.py
tomtom1=${OUT_DIR}/tomtom/tomtom_test.tsv
tomtom2=../tutorial/ExplaiNN/AI-TAC/tomtom/tomtom.tsv

$script $tomtom2 $tomtom1
