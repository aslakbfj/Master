#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_train          # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_out/train_%j_%a.log # Stdout and stderr file


source activate explainn





#### Download the AI-TAC dataset if not there
#URL=https://www.dropbox.com/s
#FILES=("r8drj2wxc07bt4j/ImmGenATAC1219.peak_matched.txt" "7mmd4v760eux755/mouse_peak_heights.csv")

# Create the directory if it doesn't exist
#mkdir -p AI-TAC

# Loop over the files
#for FILE in ${FILES[@]}; do
  # Extract the filename from the URL
#  FILENAME=$(basename $FILE)

  # Check if the file exists
#  if [ ! -f AI-TAC/$FILENAME ]; then
    # If the file doesn't exist, download it
#    wget -P AI-TAC ${URL}/$FILE
#  else
    # If the file exists, print a message and continue
#    echo "File $FILENAME already exists, skipping download."
#  fi
#done

## Other downloads are necessary! Check out tutorial slides. Two genomepy installs are necessary.

#echo "Extract the ATAC-seq peak centers"
#cut -f 2-4 AI-TAC/ImmGenATAC1219.peak_matched.txt | \
#awk '{C=$2+int(($3-$2)/2);printf $1"\t%.0f\t%.0f\n",C-1,C;}' \
#> ./AI-TAC/AI-TAC_centers.bed




#echo "Extend the centers 125 bp in each direction (size = 251 bp, as in AI-TAC)"
#PY_SCRIPT=../../scripts/utils/resize.py
#${PY_SCRIPT} -o ./AI-TAC/AI-TAC_251bp.bed \
#./AI-TAC/AI-TAC_centers.bed ./genomes/mm10/mm10.fa.sizes 251


#echo Get the FASTA sequences of the resized peaks
#bedtools getfasta -fi ./genomes/mm10/mm10.fa -fo ./AI-TAC/AI-TAC_251bp.fa \
#-bed ./AI-TAC/AI-TAC_251bp.bed


echo "Create training/validation splits (percent = 90/10)"
# Get sequences
grep -v "^>" ../tutorial/AI-TAC/AI-TAC_251bp.fa | cut -c 2- > ./AI-TAC/AI-TAC_sequences.txt
#Get peak ids
tail -n +2 ../tutorial/AI-TAC/mouse_peak_heights.csv | cut -d "," -f 1 \
> ./AI-TAC/AI-TAC_ids.txt
#Get peak heights
tail -n +2 ../tutorial/AI-TAC/mouse_peak_heights.csv | cut -d "," -f 2- | \
tr "," "\t" > ./AI-TAC/AI-TAC_heights.txt
#### Make binary peak heigths
awk -v OFS="\t" '{for(i=1; i<=NF; i++) if($i > 2) $i=1; else $i=0} 1' ./AI-TAC/AI-TAC_heights.txt > ./AI-TAC/AI-TAC_binary_heights.txt


paste -d "\t" ./AI-TAC/AI-TAC_ids.txt ./AI-TAC/AI-TAC_sequences.txt \
./AI-TAC/AI-TAC_binary_heights.txt > AI-TAC/AI-TAC_251bp.tsv
awk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.1 ? ".validation" : ".train");
print > f}' ./AI-TAC/AI-TAC_251bp.tsv



echo "Train (same parameters as in the preprint; it can take a few hours) and test"
### loss has been changed to BCE with logits
PY_SCRIPT=../../scripts/train.py
OUT_DIR=./ExplaiNN/AI-TAC
${PY_SCRIPT} -t -o ${OUT_DIR} --input-length 251 --criterion bcewithlogits \
--num-units 300 ./AI-TAC/AI-TAC_251bp.tsv.train \
--patience 5 \
--num-epochs 100 \
./AI-TAC/AI-TAC_251bp.tsv.validation
