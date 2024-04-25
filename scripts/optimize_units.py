#!/usr/bin/env python

import click
from click_option_group import optgroup
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch
from torch import nn
import tqdm
from explainn.train.train import train_explainn
from explainn.utils.tools import pearson_loss
from explainn.models.networks import ExplaiNN
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

from explainn import tools
from explainn import networks
from explainn import train
from explainn import test
from explainn import interpretation
from sklearn.metrics import average_precision_score
from sklearn import metrics
import matplotlib.pyplot as plt

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "h5",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "output",
    type=click.Path(resolve_path=True),
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
# @click.option(
#     "-o", "--output-dir",
#     help="Output directory.",
#     type=click.Path(resolve_path=True),
#     default="./",
#     show_default=True,
# )
@click.option(
    "-t", "--time",
    help="Return the program's running execution time in seconds.",
    is_flag=True,
)
@optgroup.group("ExplaiNN")
@optgroup.option(
    "--filter-size",
    help="Size of each unit's filter.",
    type=int,
    default=19,
    show_default=True,
)
@optgroup.option(
    "--input-length",
    help="Input length (for longer and shorter sequences, trim or add padding, i.e. Ns, up to the specified length).",
    type=int,
    required=True,
)
@optgroup.option(
    "--num-fc",
    help="Number of fully connected layers in each unit.",
    type=click.IntRange(0, 8, clamp=True),
    default=2,
    show_default=True,
)
@optgroup.option(
    "--num-units",
    help="Number of independent units.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--pool-size",
    help="Size of each unit's maxpooling layer.",
    type=int,
    default=7,
    show_default=True,
)
@optgroup.option(
    "--pool-stride",
    help="Stride of each unit's maxpooling layer.",
    type=int,
    default=7,
    show_default=True,
)
@optgroup.group("Optimizer")
@optgroup.option(
    "--criterion",
    help="Loss (objective) function to use. Select \"BCEWithLogits\" for binary or multi-class classification tasks (e.g. predict the binding of one or more TFs to a sequence), \"CrossEntropy\" for multi-class classification tasks wherein only one solution is possible (e.g. predict the species of origin of a sequence between human, mouse or zebrafish), \"MSE\" for regression tasks (e.g. predict probe intensity signals), \"Pearson\" also for regression tasks (e.g. modeling accessibility across 81 cell types), and \"PoissonNLL\" for modeling count data (e.g. total number of reads at ChIP-/ATAC-seq peaks).",
    type=click.Choice(["BCEWithLogits", "CrossEntropy", "MSE", "Pearson", "PoissonNLL"], case_sensitive=False),
    required=True
)
@optgroup.option(
    "--lr",
    help="Learning rate.",
    type=float,
    default=0.003,
    show_default=True,
)
@optgroup.option(
    "--optimizer",
    help="`torch.optim.Optimizer` with which to minimize the loss during training.",
    type=click.Choice(["Adam", "SGD"], case_sensitive=False),
    default="Adam",
    show_default=True,
)
@optgroup.group("Training")
@optgroup.option(
    "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--checkpoint",
    help="How often to save checkpoints (e.g. 1 means that the model will be saved after each epoch; by default, i.e. 0, only the best model will be saved).",
    type=int,
    default=0,
    show_default=True,
)
@optgroup.option(
    "--num-epochs",
    help="Number of epochs to train the model.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--patience",
    help="Number of epochs to wait before stopping training if the validation loss does not improve.",
    type=int,
    default=10,
    show_default=True,
)
@optgroup.option(
    "--rev-complement",
    help="Reverse and complement training sequences.",
    is_flag=True,
)
@optgroup.option(
    "--trim-weights",
    help="Constrain output weights to be non-negative (i.e. to ease interpretation).",
    is_flag=True,
)

def cli(**args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # retrieve the slurm job number
    out_job = args["output"]
    os.makedirs(out_job)


    
    # Parameters
    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]
    learning_rate = args["lr"]


    dataloaders, target_labels, train_out = tools.load_datas(args["h5"], batch_size,
                                                            0, True)

    target_labels = [i.decode("utf-8") for i in target_labels]

    num_cnns = args["num_units"]
    input_length = args["input_length"]
    num_classes = len(target_labels)
    filter_size = args["filter_size"]

    # cnn_deep = networks.ConvNetDeep(num_classes)
    # cnn_shallow = networks.ConvNetShallow(num_classes)
    # #danq = networks.DanQ(num_classes)
    explainn = networks.ExplaiNN(num_cnns, input_length, num_classes, filter_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(explainn.parameters(), lr=learning_rate)

    single_folder = out_job + "/single_" + str(num_cnns) + "_units"
    os.makedirs(single_folder)
    name_ind = ""


    # training an individual model with 100 units
    explainn_100, train_error, test_error = train.train_explainn(dataloaders["train"], dataloaders["valid"], explainn,
                                                    device, criterion, optimizer, num_epochs,
                                                    single_folder, name_ind, verbose=True, trim_weights=False,
                                                    patience=args["patience"], checkpoint=args["checkpoint"])
    
    tools.showPlot(train_error, test_error, title="Loss trend", save=single_folder + "/loss_trend.png")


    ###### OPTIMIZE UNITS ######

        # Code to test how performance depends of the number of units
    num_classes = len(target_labels) #number of classes

    for num_cnns in range(0,4,2):
        if num_cnns == 0:
            num_cnns = 1    

        explainn = networks.ExplaiNN(num_cnns, input_length, num_classes, filter_size).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(explainn.parameters(), lr=learning_rate)
        out_dir=out_job + "/" + str(num_cnns)
        os.makedirs(out_dir)
        model, train_error, test_error = train.train_explainn(dataloaders["train"], dataloaders["valid"], explainn,
                                                    device, criterion, optimizer, num_epochs,
                                                    out_dir,
                                                    name_ind, verbose=True, trim_weights=False,
                                                    patience=args["patience"], checkpoint=args["checkpoint"])
        
    #old dir: ExplaiNN_filters_TF_binding/ExplaiNN_TF_num_cnns_
        print("Numm_cnns: " + str(num_cnns))
        print("Min test error: " + str(np.min(test_error)))

    # testing
    auprc_perf = {}
    num_classes = len(target_labels)
    for num_cnns in range(0,4,2):
        if num_cnns == 0:
            num_cnns = 1
        
        model = networks.ExplaiNN(num_cnns, input_length, num_classes, filter_size).to(device)
        
        #load the best model
        #old dir: CAM_filters_TF_binding/CAM_TF_num_cnns_
        #find the file that contains "best" in its name in the dir os.listdir("../../../SCRATCH/AS-TAC/ExplaiNN/single_train/ExplaiNN_TF_num_cnns_"+str(num_cnns)+"/")

        weight_file = os.listdir(out_dir + "/")[0]
        
        model.load_state_dict(torch.load(out_dir + "/" + weight_file))
        model.eval()

        labels_E, outputs_E = test.run_test(model, dataloaders["test"], device)
        
        auprc_perf[num_cnns] = average_precision_score(labels_E, outputs_E)
        
    auprc_perf = pd.Series(auprc_perf)

    #plot auprc_perf with respect to the number of units and show legend and save the plot
    plt.plot(auprc_perf)
    plt.xlabel("Number of units")
    plt.ylabel("AUPRC")
    plt.legend(["AUPRC"])
    plt.savefig(single_folder + "/AUPRC_num_units.png") 



    #### PLOT FOR EACH TARGET LABEL ####


        # performances for individual TFs (classes)
    explainn_100.eval()
    labels_E, outputs_E = test.run_test(explainn_100, dataloaders["test"], device)

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
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    targets = raw_prcs_explainn.index[:63]
    AUPRC = raw_prcs_explainn.values[:63]
    ax.bar(targets,AUPRC)
    plt.savefig(single_folder + "/AUPRC_targets.png")


if __name__ == "__main__":
    cli()
