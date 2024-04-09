#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_interpret          # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_out/interpret_%j_%a.log # Stdout and stderr file


source activate explainn

OUT_DIR=$SCRATCH/AS-TAC/ExplaiNN

echo "test?"
PY_SCRIPT=../../scripts/test.py
${PY_SCRIPT} -o ${OUT_DIR} ${OUT_DIR}/model_epoch_best_6.pth \
${OUT_DIR}/parameters-train.py.json ./AS-TAC_1000bp.validation.tsv

echo "Interpret the model"
PY_SCRIPT=../../scripts/interpret.py
${PY_SCRIPT} -t -o ${OUT_DIR} --correlation 0.75 --num-well-pred-seqs 1000 \
${OUT_DIR}/model_epoch_best_6.pth ${OUT_DIR}/parameters-train.py.json \
./AS-TAC_1000bp.train.tsv

echo "Cluster the filters (i.e., remove redundancy)"
PY_SCRIPT=../../scripts/utils/meme2clusters.py
$PY_SCRIPT -c 8 -o ${OUT_DIR}/clusters ${OUT_DIR}/filters.meme

echo "Obtain a logo for each cluster in PNG format (option �-f�)"
PY_SCRIPT=../../scripts/utils/meme2logo.py
${PY_SCRIPT} -c 8 -f png -o ${OUT_DIR}/logos ${OUT_DIR}/clusters/clusters.meme


echo "Visualize the logos of CEBP and PAX clusters (i.e., highlighted in the preprint)"
PY_SCRIPT=../../scripts/utils/tomtom.py
${PY_SCRIPT} -c 8 -o ${OUT_DIR}/tomtom ${OUT_DIR}/clusters/clusters.meme \
../JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt

#zgrep -e MA0069.1 -e MA0102.4 ${OUT_DIR}/tomtom/tomtom.tsv.gz

#### More complex visualization can be achieved by using Jupyter notebooks (or similar)
