#!/usr/bin/env python

import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch

from explainn.train.train import train_explainn
from explainn.utils.tools import dna_one_hot
from explainn.models.networks import ExplaiNN
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)


@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help='Input TSV file')
@click.option('-o', '--output', type=click.Path(), required=True, help='Output file')
@click.option('-b', '--bed_file', type=click.Path(exists=True), required=True, help='Input target label file')

def tsv2h5():
    # Load the metrics into a pandas DataFrame
    
    # Load the tsv file
    df = pd.read_csv(args['input'], sep='\t', header=None)

    # Load the bed file which contains the target labels separated by a line break
    with open(args['bed_file'], 'r') as f:
        labels = f.read().splitlines()
    
    # Remove "AS-TAC-peaks/AtlanticSalmon_ATAC_" and ".mLb.clN_peaks.narrowPeak" from the strings in labels list
    labels = [label.replace("AS-TAC-peaks/AtlanticSalmon_ATAC_", "").replace(".mLb.clN_peaks.narrowPeak", "") for label in labels]


    # remove chrom ranges if there are 65 columns
    if df.shape[1] == 65:
        df = df.drop(0, axis=1)
    
    # separate sequences and binary features
    seqs = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    # one hot encode sequences
    seqs_one_hot = np.array([dna_one_hot(str(seq)) for seq in seqs])

    # split data into train, test, valid
    train_seq, test_seq, train_feat, test_feat = train_test_split(seqs, features, test_size=0.20, random_state=42)
    train_seq, valid_seq, train_feat, valid_feat = train_test_split(train_seq, train_feat, test_size=0.25, random_state=42)


    # create .h5 file
    with h5py.File(args['output'], 'w') as hf:
        hf.create_dataset('train_in', data=train_seq)
        hf.create_dataset('valid_in', data=valid_seq)
        hf.create_dataset('test_in', data=test_seq)
        hf.create_dataset('train_out', data=train_feat)
        hf.create_dataset('valid_out', data=valid_feat)
        hf.create_dataset('test_out', data=test_feat)
        hf.create_dataset('target_labels', data=labels)

if __name__ == "__main__":
    tsv2h5()