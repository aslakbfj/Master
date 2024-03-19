#!/usr/bin/env python

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get the file paths from the command line arguments
tomtom1_path = sys.argv[1]
tomtom2_path = sys.argv[2]

print(tomtom1_path)
print(tomtom2_path)
# Read the tsv files
tomtom1 = pd.read_csv(tomtom1_path, sep='\t')
tomtom2 = pd.read_csv(tomtom2_path, sep='\t')

# Merge on Target_ID
merged = pd.merge(tomtom1, tomtom2, on='Target_ID', suffixes=('_1', '_2'))

# Iterate over the columns you're interested in
for col in ['p-value', 'E-value', 'q-value', 'Overlap']:
    # Calculate the difference between the two files
    diff = merged[col + '_1'] - merged[col + '_2']

    # Create a bar chart with bars for the differences
    plt.bar(np.arange(len(diff)), diff, alpha=0.5, label='Differences')
    plt.xlabel('Target_IDs')
    plt.ylabel('Difference')

    # Color the bars based on whether the difference is positive or negative
    for i in range(len(diff)):
        if diff[i] > 0:
            plt.bar(i, diff[i], color='b')
        else:
            plt.bar(i, diff[i], color='r')
    plt.title(f'Difference in {col} between Continuous and Binary')
    # Save the plot as a png file
    plt.savefig(f'{col}_difference_bar_chart.png')

    # Clear the current figure so the plots don't overlap in the next iteration
    plt.clf()