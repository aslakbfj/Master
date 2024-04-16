#!/bin/bash
#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=zip_metrics             # Name of job
#SBATCH --mem=20G 			                # Default memory per CPU is 3GB
#SBATCH --output=./slurm_explainn/zip_metrics_%j.log # Stdout and stderr file


source activate explainn

# Awk the performance metrics in one file
echo "create all_metrics.csv"
echo "num_units,AUCROC,AUCPR" > all_metrics.csv
for dir in $SCRATCH/AS-TAC/ExplaiNN/optimize_units_${1}*
do
    num_units=$(basename $dir | cut -d'_' -f4)
    AUCROC=$(awk -F'\t' 'NR==2 {print $0}' $dir/performance-metrics.tsv)
    AUCPR=$(awk -F'\t' 'NR==3 {print $0}' $dir/performance-metrics.tsv)
    echo "$num_units,$AUCROC,$AUCPR" >> all_metrics.csv


done

#find $SCRATCH/AS-TAC/ExplaiNN/optimize_units_14452711_* -name "performance-metrics.tsv" -type f -exec zip performance-metrics.zip {} +