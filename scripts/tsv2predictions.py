#!/usr/bin/env python

import click
from click_option_group import optgroup
import json
import numpy as np
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from explainn.models.networks import ExplaiNN
from utils import (get_seqs_labels_ids, get_file_handle, get_data_loader,
                   get_device)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "model_file",
    type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "training_parameters_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "tsv_file",
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
@optgroup.group("Predict")
@optgroup.option(
    "--apply-sigmoid",
    help="Apply the logistic sigmoid function to outputs.",
    is_flag=True,
)
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
    seqs, _, ids = get_seqs_labels_ids(args["tsv_file"],
                                       args["debugging"],
                                       False,
                                       train_args["input_length"])

    ##############
    # Predict    #
    ############## 

    # Get device
    device = get_device()

    # Get model
    state_dict = torch.load(args["model_file"])
    for k in reversed(state_dict.keys()):
        num_classes = state_dict[k].shape[0]
        break

    # Get model
    m = ExplaiNN(train_args["num_units"], train_args["input_length"],
                 num_classes, train_args["filter_size"], train_args["num_fc"],
                 train_args["pool_size"], train_args["pool_stride"],
                 args["model_file"])

    # Test
    _predict(seqs, ids, num_classes, m, device, args["output_dir"],
             args["apply_sigmoid"], args["batch_size"])

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Execution time {seconds} seconds")

def _predict(seqs, ids, num_classes, model, device, output_dir="./",
             apply_sigmoid=False, batch_size=100):

    # Initialize
    idx = 0
    predictions = np.empty((len(seqs), num_classes, 4))
    model.to(device)
    model.eval()

    # Get training DataLoader
    data_loader = get_data_loader(
        seqs,
        np.array([s[::-1, ::-1] for s in seqs]),
        batch_size
    )

    with torch.no_grad():

        for fwd, rev in tqdm(iter(data_loader), total=len(data_loader),
                             bar_format=bar_format):

            # Get strand-specific predictions
            fwd = np.expand_dims(model(fwd.to(device)).cpu().numpy(), axis=2)
            rev = np.expand_dims(model(rev.to(device)).cpu().numpy(), axis=2)

            # Combine predictions from both strands
            fwd_rev = np.concatenate((fwd, rev), axis=2)
            mean_fwd_rev = np.expand_dims(np.mean(fwd_rev, axis=2), axis=2)
            max_fwd_rev = np.expand_dims(np.max(fwd_rev, axis=2), axis=2)

            # Concatenate predictions for this batch
            p = np.concatenate((fwd, rev, mean_fwd_rev, max_fwd_rev), axis=2)
            predictions[idx:idx+fwd.shape[0]] = p

            # Index increase
            idx += fwd.shape[0]

    # Apply sigmoid
    if apply_sigmoid:
        predictions = torch.sigmoid(torch.Tensor(predictions)).numpy()

    # Get predictions
    tsv_file = os.path.join(output_dir, "predictions.tsv.gz")
    if not os.path.exists(tsv_file):
        dfs = []
        for i in range(num_classes):
            p = predictions[:, i, :]
            df = pd.DataFrame(p, columns=["Fwd", "Rev", "Mean", "Max"])
            df["SeqId"] = ids
            df["Class"] = i 
            dfs.append(df)
        df = pd.concat(dfs)[["SeqId", "Class", "Fwd", "Rev", "Mean", "Max"]]
        df.reset_index(drop=True, inplace=True)
        df.to_csv(tsv_file, sep="\t", index=False)

if __name__ == "__main__":
    cli()