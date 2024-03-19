#!/bin/bash

tomtom1="$SCRATCH/ExplaiNN/tutorial/ExplaiNN/AI-TAC/tomtom/tomtom.tsv"
tomtom2="$SCRATCH/ExplaiNN/test/ExplaiNN/AI-TAC/tomtom/tomtom.tsv"


# Extract the Target_ID column, sort it and write it to a new file
awk -F'\t' 'NR>1 {print $2}' $tomtom1 | sort > targets1.txt
awk -F'\t' 'NR>1 {print $2}' $tomtom2 | sort > targets2.txt

# Find the unique Target_IDs in each file
comm -23 targets1.txt targets2.txt > unique_in_tomtom1.txt
comm -13 targets1.txt targets2.txt > unique_in_tomtom2.txt