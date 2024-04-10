#!/usr/bin/env python

import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument('metrics_csv', type=click.Path(exists=True))

def plot_metrics(metrics_csv):
    # Load the metrics into a pandas DataFrame
    df = pd.read_csv(metrics_csv)

    # Calculate the average AUCROC and AUCPR for each num_units
    df_grouped = df.groupby('num_units').mean()

    # Plot the averages
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df_grouped.index, df_grouped['AUCROC'])
    plt.xlabel('num_units')
    plt.ylabel('Average AUCROC')

    plt.subplot(1, 2, 2)
    plt.plot(df_grouped.index, df_grouped['AUCPR'])
    plt.xlabel('num_units')
    plt.ylabel('Average AUCPR')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics()