#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=explainn_ex3          # Name of job
#SBATCH --mem=30G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_out/example_3%j_%a.log # Stdout and stderr file


source activate explainn


#### AI-TAC dataset needed
AI_TAC=$SCRATCH/ExplaiNN/tutorial/AI-TAC
mm10=$SCRATCH/ExplaiNN/genomes/mm10

## Other downloads are necessary! Check out tutorial slides. Two genomepy installs are necessary.

echo "Extract the ATAC-seq peak centers"
cut -f 2-4 ${AI_TAC}/ImmGenATAC1219.peak_matched.txt | \
awk '{C=$2+int(($3-$2)/2);printf $1"\t%.0f\t%.0f\n",C-1,C;}' \
> ${AI_TAC}/AI-TAC_centers.bed




echo "Extend the centers 125 bp in each direction (size = 251 bp, as in AI-TAC)"
PY_SCRIPT=../../scripts/utils/resize.py
${PY_SCRIPT} -o ${AI_TAC}/AI-TAC_251bp.bed \
${AI_TAC}/AI-TAC_centers.bed ${mm10}/mm10.fa.sizes 251


echo Get the FASTA sequences of the resized peaks
bedtools getfasta -fi ${mm10}/mm10.fa -fo ${AI_TAC}/AI-TAC_251bp.fa \
-bed ${AI_TAC}/AI-TAC_251bp.bed


echo" Create training/validation splits (percent = 90/10)"
grep -v "^>" ${AI_TAC}/AI-TAC_251bp.fa | cut -c 2- > ${AI_TAC}/AI-TAC_sequences.txt
tail -n +2 ${AI_TAC}/mouse_peak_heights.csv | cut -d "," -f 1 \
> ${AI_TAC}/AI-TAC_ids.txt
tail -n +2 ${AI_TAC}/mouse_peak_heights.csv | cut -d "," -f 2- | \
tr "," "\t" > ${AI_TAC}/AI-TAC_heights.txt
paste -d "\t" ${AI_TAC}/AI-TAC_ids.txt ${AI_TAC}/AI-TAC_sequences.txt \
${AI_TAC}/AI-TAC_heights.txt > ${AI_TAC}/AI-TAC_251bp.tsv
awk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.1 ? ".validation" : ".train");
print > f}' ${AI_TAC}/AI-TAC_251bp.tsv



echo "Train (same parameters as in the preprint; it can take a few hours) and test"
PY_SCRIPT=../../scripts/train.py
OUT_DIR=$SCRATCH/ExplaiNN/tutorial/ExplaiNN/AI-TAC
${PY_SCRIPT} -o ${OUT_DIR} --input-length 251 --criterion Pearson \
--num-units 300 ${AI_TAC}/AI-TAC_251bp.tsv.train \
${AI_TAC}/AI-TAC_251bp.tsv.validation
PY_SCRIPT=../../scripts/test.py
${PY_SCRIPT} -o ${OUT_DIR} ${OUT_DIR}/model_epoch_best_7.pth \
${OUT_DIR}/parameters-train.py.json ${AI_TAC}/AI-TAC_251bp.tsv.validation


echo "Interpret the model.. check model name in folder and bash file"
PY_SCRIPT=../../scripts/interpret.py
${PY_SCRIPT} -t -o ${OUT_DIR} --correlation 0.75 --num-well-pred-seqs 1000 \
${OUT_DIR}/model_epoch_best_7.pth ${OUT_DIR}/parameters-train.py.json \
${AI_TAC}/AI-TAC_251bp.tsv.train

echo "Cluster the filters (i.e., remove redundancy)"
PY_SCRIPT=../../scripts/utils/meme2clusters.py
$PY_SCRIPT -c 8 -o ${OUT_DIR}/clusters ${OUT_DIR}/filters.meme

echo "Obtain a logo for each cluster in PNG format (option “-f”)"
PY_SCRIPT=../../scripts/utils/meme2logo.py
${PY_SCRIPT} -c 8 -f png -o ${OUT_DIR}/logos ${OUT_DIR}/clusters/clusters.meme


echo "Visualize the logos of CEBP and PAX clusters (i.e., highlighted in the preprint)"
PY_SCRIPT=../../scripts/utils/tomtom.py
${PY_SCRIPT} -c 8 -o ${OUT_DIR}/tomtom ${OUT_DIR}/clusters/clusters.meme \
./JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt

zgrep -e MA0069.1 -e MA0102.4 ${OUT_DIR}/tomtom/tomtom.tsv.gz

#### More complex visualization can be achieved by using Jupyter notebooks (or similar)
