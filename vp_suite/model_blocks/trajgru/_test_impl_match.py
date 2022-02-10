from copy import deepcopy
import sys

import torch

sys.path.append("")
sys.path.append("./Precipitation-Nowcasting")

import torch.nn as nn
import numpy as np

from nowcasting.models.trajGRU import TrajGRU as TheirTrajGRU
from vp_suite.model_blocks.trajgru.traj_gru import TrajGRU as OurTrajGRU
from vp_suite.utils.models import state_dicts_equal


def compare_implementations():
    r"""
    IMPORTANT:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_channel = 3
    num_filter = 64  # enc_channels
    t = 11
    b = 2
    in_h, in_w = 32, 32

    # set up original model
    print("setting up their model")
    b_h_w = (b, in_h, in_w)
    their_model: nn.Module = TheirTrajGRU(input_channel, num_filter, b_h_w).to(device)

    # set up our model
    print("setting up our model")
    our_model: nn.Module = OurTrajGRU(device, input_channel, num_filter, in_h, in_w).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    if not state_dicts_equal(their_model, our_model):
        raise AssertionError("State dicts not equal!")
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    if not state_dicts_equal(their_model, our_model, check_values=True):
        raise AssertionError("State dicts not equal!")

    # set up input
    print("setting up input")
    their_x = torch.rand(t, b, input_channel, in_h, in_w, device=device)
    our_x = their_x.clone().permute((1, 0, 2, 3, 4))

    # infer: their model
    print("infer: theirs")
    their_model.eval()
    their_out, their_next_h = their_model(their_x, seq_len=t)
#
    # infer: our model
    print("infer: ours")
    our_model.eval()
    our_out, our_next_h = our_model(our_x)
    our_out = our_out.permute((1, 0, 2, 3, 4))

    # checks
    print("check results")
    for (theirs, ours) in zip([their_out, their_next_h], [our_out, our_next_h]):
        theirs = theirs.detach().cpu().numpy()
        ours = ours.detach().cpu().numpy()
        if theirs.shape != ours.shape:
            raise AssertionError(f"Prediction shapes are not equal. "
                                 f"Theirs: {theirs.shape}, ours: {ours.shape}")
        # save_diff_hist(np.abs(theirs - ours), test_id)
        if not np.allclose(theirs, ours, rtol=0, atol=1e-4):
            raise AssertionError("Predictions are not equal.")


if __name__ == '__main__':
    compare_implementations()