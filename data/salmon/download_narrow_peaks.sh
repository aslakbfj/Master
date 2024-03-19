#!/bin/bash

# Define the URLs
urls=(
  "https://salmobase.org/datafiles/datasets/Aqua-Faang/nfcore/AtlanticSalmon/BodyMap/ATAC/Brain/results/bwa/mergedLibrary/macs/narrowPeak/"
  "https://salmobase.org/datafiles/datasets/Aqua-Faang/nfcore/AtlanticSalmon/BodyMap/ATAC/Gonad/results/bwa/mergedLibrary/macs/narrowPeak/"
  "https://salmobase.org/datafiles/datasets/Aqua-Faang/nfcore/AtlanticSalmon/BodyMap/ATAC/Liver/results/bwa/mergedLibrary/macs/narrowPeak/"
  "https://salmobase.org/datafiles/datasets/Aqua-Faang/nfcore/AtlanticSalmon/BodyMap/ATAC/Muscle/results/bwa/mergedLibrary/macs/narrowPeak/"
)
tissues=(
  "Brain"
  "Gonad"
  "Liver"
  "Muscle"
)

# Loop through the URLs and download the files
for i in "${!urls[@]}"; do
  wget -r -np -nH --cut-dirs=13 -A .narrowPeak -X "${urls[$i]}qc","${urls[$i]}consensus" -P ./narrow_peak "${urls[$i]}"
done

rm -r ./narrow_peak/qc
rm -r ./narrow_peak/consensus