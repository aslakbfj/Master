#!/usr/bin/env python

import click
from click_option_group import optgroup
import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch

from explainn.interpretation.interpretation import get_explainn_predictions
from explainn.models.networks import ExplaiNN
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "model_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "training_parameters_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "test_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-c", "--cpu-threads",
    help="Number of CPU threads to use.",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "-d", "--debugging",
    help="Debugging mode.",
    is_flag=True,
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True,
)
@click.option(
    "-t", "--time",
    help="Return the program's running execution time in seconds.",
    is_flag=True,
)
@optgroup.group("Test")
@optgroup.option(
    "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)

def cli(**args):

    # Start execution
    start_time = time.time()

    # Initialize
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Save exec. parameters as JSON
    json_file = os.path.join(args["output_dir"],
                             f"parameters-{os.path.basename(__file__)}.json")
    handle = get_file_handle(json_file, "wt")
    handle.write(json.dumps(args, indent=4, sort_keys=True))
    handle.close()

    ##############
    # Load Data  #
    ##############

    # Load training parameters
    handle = get_file_handle(args["training_parameters_file"], "rt")
    train_args = json.load(handle)
    handle.close()
    if "training_parameters_file" in train_args: # i.e. for fine-tuned models
        handle = get_file_handle(train_args["training_parameters_file"], "rt")
        train_args = json.load(handle)
        handle.close()

    # Get test sequences and labels
    seqs, labels, _ = get_seqs_labels_ids(args["test_file"],
                                          args["debugging"],
                                          False,
                                          train_args["input_length"])

    ##############
    # Test       #
    ############## 

    # Infer input type, and the number of classes
    num_classes = labels[0].shape[0]
    if np.unique(labels[:, 0]).size == 2:
        input_type = "binary"
    else:
        input_type = "non-binary"

    #Print input type
    print("Input type is: ")
    print(input_type)
    # Get device
    device = get_device()

    # Get model
    m = ExplaiNN(train_args["num_units"], train_args["input_length"],
                 num_classes, train_args["filter_size"], train_args["num_fc"],
                 train_args["pool_size"], train_args["pool_stride"],
                 args["model_file"])


    # Get training DataLoader
    data_loader = get_data_loader(seqs, labels, batch_size)

if __name__ == "__main__":
    cli()