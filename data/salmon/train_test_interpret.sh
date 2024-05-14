#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=train_explainn           # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_train_test_interpret/job%j.log # Stdout and stderr file


source activate explainn
# num-units as argument!
DATA_DIR=$SCRATCH/AS-TAC
TRAIN_SCRIPT=../../scripts/train.py
TEST_SCRIPT=../../scripts/test.py

#/mnt/SCRATCH/asfj/AS-TAC/AS-TAC_21_25_train.tsv
TRAIN_TSV=${DATA_DIR}/AS-TAC_noMuscle_21_25_train.tsv
TEST_TSV=${DATA_DIR}/AS-TAC_noMuscle_21_25_test.tsv

# Retrieve which tsv the model was trained on
# i.e. 21_25 and so on, where chrom 21 and 25 are test set
file_name=$(basename "$TRAIN_TSV")  # Get the file name from the path
prefix_removed=${file_name#*_}  # Remove the prefix before the first underscore
TSV_VARIANT=${prefix_removed%_*}  # Remove the suffix after the last underscore


OUT_DIR="$SCRATCH/AS-TAC/ExplaiNN/single_train/${1}_units_${TSV_VARIANT}"


# withdraw 21_25 from TRAIN_TSV

echo "Train (same parameters as in the preprint; it can take a few hours) and test"

${TRAIN_SCRIPT} -o ${OUT_DIR} --input-length 1000 --criterion bcewithlogits \
--patience 10 \
--num-epochs 200 \
--lr 0.005 \
--batch-size 100 \
--num-units ${1} ${TRAIN_TSV} ${TEST_TSV}

echo "training done"

echo "testing..."

# get the best model
PTH_FILE=$(ls ${OUT_DIR}/*.pth)
# test the model
${TEST_SCRIPT} -o ${OUT_DIR} ${PTH_FILE} \
${OUT_DIR}/parameters-train.py.json ${TEST_TSV}


echo "Interpret the model"
INTERPRET_SCRIPT=../../scripts/interpret.py

${INTERPRET_SCRIPT} -t -o ${OUT_DIR} --correlation 0.75 --num-well-pred-seqs 1000 \
${PTH_FILE} ${OUT_DIR}/parameters-train.py.json \
${TRAIN_TSV}


# Uncomment for clustering
# echo "Cluster the filters (i.e., remove redundancy)"
# PY_SCRIPT=../../scripts/utils/meme2clusters.py
# $PY_SCRIPT -c 8 -o ${OUT_DIR}/clusters ${OUT_DIR}/filters.meme

# echo "Obtain a logo for each cluster in PNG format (option -f)"
# PY_SCRIPT=../../scripts/utils/meme2logo.py
# ${PY_SCRIPT} -c 8 -f png -o ${OUT_DIR}/logos ${OUT_DIR}/clusters/clusters.meme


echo "Visualize the logos of CEBP and PAX clusters (i.e., highlighted in the preprint)"

PY_SCRIPT=../../scripts/utils/tomtom.py
#wget https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt -P ../JASPAR/
${PY_SCRIPT} -c 8 -o ${OUT_DIR}/tomtom ${OUT_DIR}/filters.meme \
../JASPAR/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
#${OUT_DIR}/clusters/clusters.meme \ #

sbatch plot_metrics.sh ${1}_units_${TSV_VARIANT} bed_list_noMuscle.txt
#z grep -e MA0069.1 -e MA0102.4 ${OUT_DIR}/tomtom/tomtom.tsv.gz

#### More complex visualization can be achieved by using Jupyter notebooks (or similar)

