#!/usr/bin/env python

import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
from explainn import tools
from explainn import networks
from explainn import train
from explainn import test
from explainn import interpretation
import click
from click_option_group import optgroup
import torch
import os
import json
from torch import nn
from sklearn.metrics import average_precision_score
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

@click.command()
@click.option(
    "-i", "--model-dir",
    type=click.Path(), required=True,
    help='Output plot file')

@click.option(
    "-d", "--debugging",
    help="Debugging mode.",
    is_flag=True,
)

@click.option(
    "-o", "--output",
    help="output model name.")

def cli(**args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # List all the files in the model directory
    model_dir_list = os.listdir(model_dir)
    output = args["output"]

    # Load training parameters
    handle = get_file_handle(model_dir + "/parameters-train.py.json", "rt")
    train_args = json.load(handle)
    handle.close()
    if "training_parameters_file" in train_args: # i.e. for fine-tuned models
        handle = get_file_handle(train_args["training_parameters_file"], "rt")
        train_args = json.load(handle)
        handle.close()
        # Else, show an error message
    else:
        print("No training_parameters_file key in the model directory.")
        sys.exit(1)

    # find the .pth file in weight_file list
    for file in model_dir_list:
        if file.endswith(".pth"):
            weight_file = file
            break

    # Get model parameters
    num_cnns = train_args['num_units']
    input_length = train_args['input_length']
    num_classes = 63
    filter_size = train_args['filter_size']

    # Define model
    explainn = networks.ExplaiNN(num_cnns, input_length, num_classes, filter_size).to(device)
    explainn.load_state_dict(torch.load(model_dir + "/" + weight_file))
    explainn.eval()



    # Get test sequences and labels
    seqs, labels, _ = get_seqs_labels_ids(train_args["validation_file"],
                                          args["debugging"],
                                          False,
                                          train_args["input_length"])
    

    # Get training DataLoader
    data_loader = get_data_loader(seqs, labels, train_args["batch_size"])

    

    labels_E, outputs_E = test.run_test(explainn, data_loader, device)

    no_skill_probs = [0 for _ in range(len(labels_E[:,0]))]
    ns_fpr, ns_tpr, _ = metrics.roc_curve(labels_E[:,0], no_skill_probs)

    roc_aucs = {}
    raw_aucs = {}
    roc_prcs = {}
    raw_prcs = {}
    for i in range(len(target_labels)):
        nn_fpr, nn_tpr, threshold = metrics.roc_curve(labels_E[:,i], outputs_E[:,i])
        roc_auc_nn = metrics.auc(nn_fpr, nn_tpr)
        
        precision_nn, recall_nn, thresholds = metrics.precision_recall_curve(labels_E[:,i], outputs_E[:,i])
        pr_auc_nn = metrics.auc(recall_nn, precision_nn)
        
        roc_aucs[target_labels[i]] = nn_fpr, nn_tpr
        raw_aucs[target_labels[i]] = roc_auc_nn
        
        roc_prcs[target_labels[i]] = recall_nn, precision_nn
        raw_prcs[target_labels[i]] = pr_auc_nn
        
    raw_prcs_explainn = pd.Series(raw_prcs)
    raw_aucs_explainn = pd.Series(raw_aucs)
    # AUPRC
    # The TFs index object has the target labels. And within the 63 labes, I want to color by labels containing the words "Brain", "Liver", "Gonad", "Muscle", "MidSomitogenesis", "LateSomitogenesis", "Lateblastulation", "EarlySomitogenesis_R2"
    # I will use the seaborn library to color the bars

    # make a new column that is the first word before "_" in the rownames of the Series object turned into a df
    df = raw_prcs_explainn.reset_index()
    df['tissue'] = df['index'].str.split('_').str[0]

    df_aucs = raw_aucs_explainn.reset_index()
    df_aucs['tissue'] = df_aucs['index'].str.split('_').str[0]

    # get the classes, which are raw_prcs_explainn.index
    classes = raw_prcs_explainn.index
    AUPRC = raw_prcs_explainn.values
    #make the sns.barplot bigger

    sns.barplot(x=classes, y=AUPRC, hue=df['tissue'], palette="tab10", title="AUPRC using "+ output)
    plt.savefig(output +'.png')

if __name__ == "__main__":
    cli()