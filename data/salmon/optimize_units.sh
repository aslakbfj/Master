#!/bin/bash
#SBATCH --job-name=optimize_units
#SBATCH --output=optimize_units_%j_%a.out
#SBATCH --error=optimize_units_%j_%a.err
#SBATCH --array=1-10%10
#SBATCH --time=48:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_explainn/optimize_units%j_%a.log # Stdout and stderr file
#SBATCH --error=./slurm_explainn/optimize_units%j_%a.err # Stdout and stderr file

source activate explainn
# Liste over num-units verdier du vil teste
# Adjust this list based on the number of array jobs you want to run
NUM_UNITS_LIST=(100 150 200 250 300 350 400 450 500 600)

# Delen av koden som ikke endrer seg
INPUT_LENGTH=1000
CRITERION=bcewithlogits
PATIENCE=6
NUM_EPOCHS=100
TRAIN_TSV=AS-TAC_1000bp.train.tsv
VALIDATION_TSV=AS-TAC_1000bp.validation.tsv
TRAIN_SCRIPT=../../scripts/train.py
TEST_SCRIPT=../../scripts/test.py

# Get the correct num-units value based on the array job index
NUM_UNITS=${NUM_UNITS_LIST[$SLURM_ARRAY_TASK_ID-1]}
echo "num-units: ${NUM_UNITS}"

# Create and set output directory
OUT_DIR="$SCRATCH/AS-TAC/ExplaiNN/omptimize_units_${SLURM_ARRAY_JOB_ID}_${NUM_UNITS}"
mkdir -p ${OUT_DIR}

# Run script
${TRAIN_SCRIPT} -o ${OUT_DIR} --input-length ${INPUT_LENGTH} --criterion ${CRITERION} \
--patience ${PATIENCE} \
--num-epochs ${NUM_EPOCHS} \
--num-units ${NUM_UNITS} ${TRAIN_TSV} \
${VALIDATION_TSV}

echo "training done"

echo "test"

#get the best model
PTH_FILE=$(ls ${OUT_DIR}/*.pth)
${TEST_SCRIPT} -o ${OUT_DIR} ${PTH_FILE} \
${OUT_DIR}/parameters-train.py.json ./AS-TAC_1000bp.validation.tsv

echo "should be done now"
echo "contents of OUT_DIR: $(ls ${OUT_DIR})"