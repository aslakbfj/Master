#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=plot_metrics             # Name of job
#SBATCH --mem=20G 			                # Default memory per CPU is 3GB
#SBATCH --output=./slurm_explainn/plot_metrics_%j.log # Stdout and stderr file


source activate explainn


# Awk the performance metrics in one file
echo "create metrics.csv"
echo "num_units,AUCROC,AUCPR" > metrics.csv
# input argument the job ID in order to choose comparable models!
for dir in $SCRATCH/AS-TAC/ExplaiNN/optimize_units_${1}/*
do
    num_units=$(basename $dir | cut -d'_' -f4)
    AUCROC=$(awk -F'\t' 'NR==2 {print $2}' $dir/performance-metrics.tsv)
    AUCPR=$(awk -F'\t' 'NR==3 {print $2}' $dir/performance-metrics.tsv)
    echo "$num_units,$AUCROC,$AUCPR" >> metrics.csv
done

echo "plot the metrics"
# Plot the performance metrics
PY_SCRIPT=../../scripts/plot_metrics.py
${PY_SCRIPT} -i metrics.csv -o ./optimize_units_metrics_${1}.png
