#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=compare_tomtom             # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --output=./slurm/compare_tomtom%j.log       # Stdout and stderr file


# source activate explainn

#make a file listing the path of all tomtom/tomtom.tsv files in the different folders in "../../SCRATCH/AS-TAC/single_train/" that contain "merged" in them


# list all folders in "../../../../SCRATCH/AS-TAC/single_train/" that contain "merged" in them to a file, then append "tomtom/tomtom.tsv" to each line in the file
single_train="../../../../SCRATCH/AS-TAC/ExplaiNN/single_train/"
merged_tomtom="../../../../SCRATCH/AS-TAC/ExplaiNN/merged_tomtom"
mkdir -p $merged_tomtom
# sort the lines in alphabetical order
find $single_train -type d -name "*merged*" | sed 's/$/\/tomtom\/tomtom.tsv/' | sort > ${merged_tomtom}/merged_tomtom_files.txt
#find $single_train -type d -name "*merged*" | sed 's/$/\/tomtom\/tomtom.tsv/' > ${merged_tomtom}/merged_tomtom_files.txt

