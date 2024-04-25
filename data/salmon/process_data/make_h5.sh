#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=process_data             # Name of job
#SBATCH --mem=60G 			                # Default memory per CPU is 3GB
#SBATCH --output=./slurm/tsv2h5%j.log       # Stdout and stderr file


source activate explainn

echo "num-units as argument!"
TSV_SCRIPT=../../../scripts/tsv2h5.py
OUT_DIR=$SCRATCH/AS-TAC
BED_LIST=$SCRATCH/downloads/genomes/salmon/bed_list_test.txt
TSV_FILE=$SCRATCH/AS-TAC/AS-TAC_1000bp.tsv

echo "running tsv to h5 script"
$TSV_SCRIPT -o $OUT_DIR -b $BED_LIST -i $TSV_FILE
echo "done"