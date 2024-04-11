#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=plot_metrics             # Name of job
#SBATCH --mem=20G 			                # Default memory per CPU is 3GB
#SBATCH --output=./slurm_explainn/plot_metrics%j_%a.log # Stdout and stderr file


source activate explainn


# Awk the performance metrics in one file
#awk 'FNR==1 && NR!=1{next;}{print}' ${OUT_DIR}/*.metrics > ${OUT_DIR}/all_metrics.tsv


echo "num_units,AUCROC,AUCPR" > metrics.csv
for dir in $SCRATCH/AS-TAC/ExplaiNN/omptimize_units_${1}*
do
    num_units=$(basename $dir | cut -d'_' -f3)
    AUCROC=$(awk -F'\t' 'NR==2 {print $2}' $dir/performance-metrics.tsv)
    AUCPR=$(awk -F'\t' 'NR==3 {print $2}' $dir/performance-metrics.tsv)
    echo "$num_units,$AUCROC,$AUCPR" >> metrics.csv
done


# Plot the performance metrics
PY_SCRIPT=../../scripts/plot_metrics.py
${PY_SCRIPT} -i metrics.csv -o ./omptimize_units_metrics.png
