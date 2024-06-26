# =============================================================================
# IMPORTS
# =============================================================================
import torch.nn as nn
import torch
import numpy as np

# =============================================================================
# FUNCTIONS
# =============================================================================

def run_transform(model, data_loader, device, isSigmoid=False):
    """

    :param model: ExplaiNN model
    :param data_loader: torch DataLoader, data loader with the sequences of interest
    :param device: current available device ('cuda:0' or 'cpu')
    :param isSigmoid: boolean, True if the model output is binary
    :return:
    """
    running_outputs = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for seq in data_loader:
            seq = seq.to(device)
            out = model(seq)
            out = out.detach().cpu()
            if isSigmoid:
                out = sigmoid(out)
            running_outputs.extend(out.numpy())
    return np.array(running_outputs)
