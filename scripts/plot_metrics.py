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

    # Create a new figure
    plt.figure()

    # Plot AUCROC
    plt.plot(df['num_units'], df['AUCROC'], label='AUCROC')

    # Plot AUCPR
    plt.plot(df['num_units'], df['AUCPR'], label='AUCPR')

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel('num_units')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig(output)

if __name__ == "__main__":
    plot_metrics()