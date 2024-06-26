#!/usr/bin/env python

import click
import pandas as pd
import matplotlib.pyplot as plt


@click.command()
@click.option('-i', '--input', 'metrics_csv', type=click.Path(exists=True), required=True, help='Input CSV file')
@click.option('-o', '--output', type=click.Path(), required=True, help='Output plot file')

def plot_metrics(metrics_csv, output):
    # Load the metrics into a pandas DataFrame
    df = pd.read_csv(metrics_csv)
    df = df.dropna(how='all')
    df = df.sort_values(by=['num_units'])
    # Create a new figure
    plt.figure()

    # Plot AUCROC
    plt.plot(df['num_units'], df['AUCROC'], label='AUCROC')

    # Plot AUCPR
    plt.plot(df['num_units'], df['AUCPR'], label='AUCPR')
    plt.xlim(0,200)
    plt.ylim(0,0.8)
    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel('num_units')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig(output)


with open("../SCRATCH/AS-TAC/bed_list_test.txt", 'r') as f:
target_labels = f.read().splitlines()
# get the number of classes from target_labels
num_classes = len(target_labels)
if __name__ == "__main__":
    plot_metrics()