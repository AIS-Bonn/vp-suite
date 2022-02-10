import argparse
import os
from copy import deepcopy
import sys

import torch

sys.path.append("")
sys.path.append("./ConvLSTM_pytorch")

import torch.nn as nn
import numpy as np

from convlstm import ConvLSTM as TheirConvLSTM
from conv_lstm import ConvLSTM as OurConvLSTM
from vp_suite.utils.models import state_dicts_equal
from vp_suite.utils.visualization import save_diff_hist

def compare_implementations():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 3
    hidden_dim = 64
    kernel_size = (3, 3)
    num_layers = 3
    batch_first = True
    return_all_layers = True

    # set up original model
    print("setting up their model")
    their_model: nn.Module = TheirConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=batch_first,
                                           return_all_layers=return_all_layers).to(device)

    # set up our model
    print("setting up our model")
    our_model: nn.Module = OurConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=batch_first,
                                       return_all_layers=return_all_layers).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    assert state_dicts_equal(their_model, our_model), "State dicts not equal!"
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    assert state_dicts_equal(their_model, our_model, check_values=True), "State dicts not equal!"

    # set up input
    print("setting up input")
    their_x = torch.rand(2, 11, 3, 67, 83, device=device)
    our_x = their_x.clone()

    # infer: their model
    print("infer: theirs")
    their_model.eval()
    their_out, their_state = their_model(their_x)
    their_h, their_c = list(zip(*their_state))
    all_their_outputs = their_out + list(their_h) + list(their_c)
#
    # infer: our model
    print("infer: ours")
    our_model.eval()
    our_out, our_state = our_model(our_x)
    our_h, our_c = list(zip(*our_state))
    all_our_outputs = our_out + list(our_h) + list(our_c)

    # checks
    print("check results")
    for (theirs, ours) in zip(all_their_outputs, all_our_outputs):
        theirs = theirs.detach().cpu().numpy()
        ours = ours.detach().cpu().numpy()
        assert theirs.shape == ours.shape, f"Prediction shapes are not equal. " \
                                           f"Theirs: {theirs.shape}, ours: {ours.shape}"
        # save_diff_hist(np.abs(theirs - ours), test_id)
        assert np.allclose(theirs, ours, rtol=0, atol=1e-4), "Predictions are not equal."


if __name__ == '__main__':
    compare_implementations()