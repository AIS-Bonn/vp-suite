from copy import deepcopy
import sys, os

import torch
import torch.nn as nn
import numpy as np

from vp_suite.model_blocks import ConvLSTM as OurConvLSTM
from vp_suite.utils.models import state_dicts_equal

REFERENCE_GIT_URL = "https://github.com/Hzzone/Precipitation-Nowcasting.git"
REPO_DIR = "Precipitation-Nowcasting"


# noinspection PyUnresolvedReferences
def test_impl():

    # cfg needs to be imported due to circular import in their code, however it is not loadable by default due to
    # faulty assertion statements -> Remove 'assert' statements from config file so that it actually gets loaded.
    cfg_module_fp = os.path.join(sys.path[0], "nowcasting/config.py")
    with open(cfg_module_fp, 'r') as cfg_module_file:
        lines = cfg_module_file.readlines()
    with open(cfg_module_fp, 'w') as cfg_module_file:
        for line in lines:
            if "assert" not in line:
                cfg_module_file.write(line)

    from nowcasting.models.convLSTM import ConvLSTM as TheirConvLSTM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_channel = 3
    num_filter = 64  # enc_channels
    b = 2
    state_h, state_w = 32, 32
    kernel_size = (3, 3)
    stride = 1
    padding = 1

    # set up original model
    print("setting up their model")
    b_h_w = (b, state_h, state_w)
    their_model: nn.Module = TheirConvLSTM(input_channel, num_filter, b_h_w, kernel_size, stride=stride,
                                           padding=padding).to(device)

    # set up our model
    print("setting up our model")
    our_model: nn.Module = OurConvLSTM(device, input_channel, num_filter, state_h, state_w, kernel_size,
                                       stride=stride, padding=padding).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    if not state_dicts_equal(their_model, our_model):
        raise AssertionError("State dicts not equal!")
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    for weight_matrix in ["Wci", "Wcf", "Wco"]:
        their_w = getattr(their_model, weight_matrix)
        our_w = getattr(our_model, weight_matrix)
        if their_w.data.shape != our_w.data.shape:
            raise AssertionError(f"Parameter shapes for '{weight_matrix}' not equal!")
        our_w.data = their_w.data.clone()
        if our_w.data.ne(their_w.data).sum() > 0:
            raise AssertionError(f"Values for parameter '{weight_matrix}' not equal!")
    if not state_dicts_equal(their_model, our_model, check_values=True):
        raise AssertionError("State dicts not equal!")

    # set up input
    print("setting up input")
    their_x = torch.rand(11, b, input_channel, state_h, state_w, device=device)
    our_x = their_x.clone().permute((1, 0, 2, 3, 4))

    # infer: their model
    print("infer: theirs")
    their_model.eval()
    their_out, (their_h, their_c) = their_model(their_x)
#
    # infer: our model
    print("infer: ours")
    our_model.eval()
    our_out, (our_h, our_c) = our_model(our_x, states=None, seq_len=5)
    our_out = our_out.permute((1, 0, 2, 3, 4))

    # checks
    print("check results")
    for (theirs, ours) in zip([their_out, their_h, their_c], [our_out, our_h, our_c]):
        theirs = theirs.detach().cpu().numpy()
        ours = ours.detach().cpu().numpy()
        if theirs.shape != ours.shape:
            raise AssertionError(f"Prediction shapes are not equal. "
                                 f"Theirs: {theirs.shape}, ours: {ours.shape}")
        # save_arr_hist(np.abs(theirs - ours), test_id)
        if not np.allclose(theirs, ours, rtol=0, atol=1e-4):
            raise AssertionError("Predictions are not equal.")
