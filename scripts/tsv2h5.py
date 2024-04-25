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
import os
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

def cli(**args):
    # Load the metrics into a pandas DataFrame
    
    # Load the tsv file
    df = pd.read_csv(args['input'], sep='\t', header=None)

    # Load the bed file which contains the target labels separated by a line break
    with open(args['bed_file'], 'r') as f:
        labels = f.read().splitlines()
    
    # Remove "AS-TAC-peaks/AtlanticSalmon_ATAC_" and ".mLb.clN_peaks.narrowPeak" from the strings in labels list
    labels = [label.replace("AS-TAC-peaks/AtlanticSalmon_ATAC_", "").replace(".mLb.clN_peaks.narrowPeak", "") for label in labels]



    ########## make a subset of the data that starts with "21" or "25" ###############

    df_21_25 = df[df[0].str.startswith('21') | df[0].str.startswith('25')]
    # make a subset of the rest of the data
    df_excluded_21_25 = df[~(df[0].str.startswith('21') | df[0].str.startswith('25'))]

    # remove chrom ranges if there are 65 columns
    if df_21_25.shape[1] == 65:
        df_21_25 = df_21_25.drop(0, axis=1)
    # remove chrom ranges if there are 65 columns
    if df_excluded_21_25.shape[1] == 65:
        df_excluded_21_25 = df_excluded_21_25.drop(0, axis=1)
    

    ###################################################################
    ########## MAKE A 5 % SPLIT WITHOUT SEPARATING 21/25    ###########
    ##########                                              ###########
    ###################################################################


    # remove chrom ranges if there are 65 columns
    if df.shape[1] == 65:
        df = df.drop(0, axis=1)
    
    # separate sequences and binary features
    seqs = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    #remove df to save memory
    del df

    # one hot encode sequences
    seqs = np.array([dna_one_hot(str(seq)) for seq in seqs])

    # split data into train, test, valid
    train_seq_05, valid_seq_05, train_feat_05, valid_feat_05 = train_test_split(seqs, features, test_size=0.05, random_state=42)
    train_seq_05, test_seq_05, train_feat_05, test_feat_05 = train_test_split(train_seq_05, train_feat_05, test_size=0.052, random_state=42)

    # create .h5 file
    with h5py.File(args['output'] + "AS-TAC_05.h5", 'w') as hf:
        hf.create_dataset('train_in', data=train_seq_05)
        hf.create_dataset('valid_in', data=valid_seq_05)
        hf.create_dataset('test_in', data=test_seq_05)
        hf.create_dataset('train_out', data=train_feat_05)
        hf.create_dataset('valid_out', data=valid_feat_05)
        hf.create_dataset('test_out', data=test_feat_05)
        hf.create_dataset('target_labels', data=labels)

    # Remove train_05, valid_05, test_05 to save memory
    del train_seq_05, valid_seq_05, test_seq_05, train_feat_05, valid_feat_05, test_feat_05


    ###################################################################
    ########## MAKE A 10 % SPLIT WITHOUT SEPARATING 21/25   ###########
    ##########                                              ###########
    ###################################################################

    # split data into train_10, test_10, valid_10 with a 10 % split
    train_seq_10, valid_seq_10, train_feat_10, valid_feat_10 = train_test_split(seqs, features, test_size=0.1, random_state=42)
    train_seq_10, test_seq_10, train_feat_10, test_feat_10 = train_test_split(train_seq_10, train_feat_10, test_size=0.1, random_state=42)

    # create .h5 file
    with h5py.File(args['output'] + "AS-TAC_10.h5", 'w') as hf:
        hf.create_dataset('train_in', data=train_seq_10)
        hf.create_dataset('valid_in', data=valid_seq_10)
        hf.create_dataset('test_in', data=test_seq_10)
        hf.create_dataset('train_out', data=train_feat_10)
        hf.create_dataset('valid_out', data=valid_feat_10)
        hf.create_dataset('test_out', data=test_feat_10)
        hf.create_dataset('target_labels', data=labels)


    # Remove train_10, valid_10, test_10 to save memory
    del train_seq_10, valid_seq_10, test_seq_10, train_feat_10, valid_feat_10, test_feat_10
    del seqs, features

    ###################################################################
    ##########  MAKE A 10 % SPLIT WITH 21 25 SEPARATED       ###########
    ##########                                              ###########
    ###################################################################

    # Make a new df_21_25_10 from where you add a random 5 % of the rows of df_excluded_21_25 to df_21_25, and remove those rows from df_excluded_21_25 to a new df with _10. use pd.concat
    sample = df_excluded_21_25.sample(frac=0.05)
    df_21_25_10 = pd.concat([df_21_25, sample])
    df_excluded_21_25_10 = df_excluded_21_25.drop(sample.index)
    del sample
    # separate sequences and binary features
    seqs_21_25_10 = df_21_25_10.iloc[:, 0]
    features_21_25_10 = df_21_25_10.iloc[:, 1:]
    seqs_excluded_21_25_10 = df_excluded_21_25_10.iloc[:, 0]
    features_excluded_21_25_10 = df_excluded_21_25_10.iloc[:, 1:]

    # one hot encode sequences
    seqs_21_25_10 = np.array([dna_one_hot(str(seq)) for seq in seqs_21_25_10])
    seqs_excluded_21_25_10 = np.array([dna_one_hot(str(seq)) for seq in seqs_excluded_21_25_10])

    # split data into train_21_25_10, test_21_25_10
    train_seq_21_25_10, test_seq_21_25_10, train_feat_21_25_10, test_feat_21_25_10 = train_test_split(seqs_excluded_21_25_10, features_excluded_21_25_10, test_size=0.1, random_state=42)
    valid_seq_21_25_10, valid_feat_21_25_10 = seqs_21_25_10, features_21_25_10

    # create .h5 file
    with h5py.File(args['output'] + "AS-TAC_21_25_10.h5", 'w') as hf:
        hf.create_dataset('train_in', data=train_seq_21_25_10)
        hf.create_dataset('valid_in', data=valid_seq_21_25_10)
        hf.create_dataset('test_in', data=test_seq_21_25_10)
        hf.create_dataset('train_out', data=train_feat_21_25_10)
        hf.create_dataset('valid_out', data=valid_feat_21_25_10)
        hf.create_dataset('test_out', data=test_feat_21_25_10)
        hf.create_dataset('target_labels', data=labels)


    # Remove train_21_25_10, valid_21_25_10, test_21_25_10 to save memory
    del train_seq_21_25_10, valid_seq_21_25_10, test_seq_21_25_10, train_feat_21_25_10, valid_feat_21_25_10, test_feat_21_25_10

    ########### MAKE A 5 % SPLIT WITH 21 25 SEPARATED ###########

    # separate sequences and binary features
    seqs_21_25 = df_21_25.iloc[:, 0]
    features_21_25 = df_21_25.iloc[:, 1:]
    seqs_excluded_21_25 = df_excluded_21_25.iloc[:, 0]
    features_excluded_21_25 = df_excluded_21_25.iloc[:, 1:]

    # one hot encode sequences
    seqs_21_25 = np.array([dna_one_hot(str(seq)) for seq in seqs_21_25])
    seqs_excluded_21_25 = np.array([dna_one_hot(str(seq)) for seq in seqs_excluded_21_25])

    # split data into train_21, test_21
    train_seq_21, test_seq_21, train_feat_21, test_feat_21 = train_test_split(seqs_excluded_21_25, features_excluded_21_25, test_size=0.05, random_state=42)
    valid_seq_21, valid_feat_21 = seqs_21_25, features_21_25

    # create .h5 file
    with h5py.File(args['output'] + "AS-TAC_21_25_05.h5", 'w') as hf:
        hf.create_dataset('train_in', data=train_seq_21)
        hf.create_dataset('valid_in', data=valid_seq_21)
        hf.create_dataset('test_in', data=test_seq_21)
        hf.create_dataset('train_out', data=train_feat_21)
        hf.create_dataset('valid_out', data=valid_feat_21)
        hf.create_dataset('test_out', data=test_feat_21)
        hf.create_dataset('target_labels', data=labels)

    # Remove train_21, valid_21, test_21 to save memory which is quite unnecessary as we are done (Y)
    del train_seq_21, valid_seq_21, test_seq_21, train_feat_21, valid_feat_21, test_feat_21
    

if __name__ == "__main__":
    cli()