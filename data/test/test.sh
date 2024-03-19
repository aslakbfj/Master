#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_interpret          # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_out/interpret_%j_%a.log # Stdout and stderr file


source activate explainn


OUT_DIR=./ExplaiNN/AI-TAC
echo "removing old performance-metrics.tsv and json."
rm ${OUT_DIR}/performance-metrics.tsv

echo "test, remember to use the intended model.pth"
PY_SCRIPT=../../scripts/test.py
${PY_SCRIPT} -t -o ${OUT_DIR} ${OUT_DIR}/model_epoch_best_6.pth \
${OUT_DIR}/parameters-train.py.json ./AI-TAC/AI-TAC_251bp.tsv.validation
# output: (folder “./ExplaiNN/AI-TAC/”) losses.tsv, model_epoch_best_8.pth (name can differ), parameters-train.py.json, performance-metrics.tsv
