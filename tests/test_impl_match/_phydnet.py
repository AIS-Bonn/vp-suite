from copy import deepcopy
import sys, os

import torch
import numpy as np

from vp_suite.models import PhyDNet as OurModel
from vp_suite.utils.models import state_dicts_equal

REFERENCE_GIT_URL = OurModel.CODE_REFERENCE
REPO_DIR = "PhyDNet"


# noinspection PyUnresolvedReferences
def test_impl():

    from models.models import ConvLSTM as TheirConvLSTM, PhyCell as TheirPhyCell, EncoderRNN as TheirEncoderRNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    context_frames, pred_frames = 10, 10
    c, h, w = 1, 64, 64
    phydnet_config = {
        "img_shape": (c, h, w),
        "action_size": 0,
        "tensor_value_range": [0.0, 1.0],
        "moment_loss_scale": 1.0,
        "phycell_n_layers": 1,
        "phycell_channels": 49,
        "phycell_kernel_size": (7, 7),
        "convlstm_n_layers": 3,
        "convlstm_hidden_dims": [128, 128, 64],
        "convlstm_kernel_size": (3, 3),
    }

    # set up original models
    print("setting up their models")

    their_phycell = TheirPhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                                 n_layers=1, kernel_size=(7, 7), device=device)
    their_convlstm = TheirConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                                   n_layers=3, kernel_size=(3, 3), device=device)
    their_model = TheirEncoderRNN(their_phycell, their_convlstm, device).to(device)

    # set up our models
    print("setting up our models")
    our_model = OurModel(device, **phydnet_config).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    if not state_dicts_equal(their_model, our_model, verbose=True):
        raise AssertionError("State dicts not equal!")
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    if not state_dicts_equal(their_model, our_model, check_values=True):
        raise AssertionError("State dicts not equal!")

    # set up input
    print("setting up input")
    their_x = torch.rand(batch_size, context_frames, c, h, w, device=device)
    our_x = their_x.clone()

    # infer: their model
    print("infer: theirs")
    their_model.eval()
    their_out = their_model_forward(their_model, their_x, context_frames, pred_frames)
    print(their_out.shape)

    # infer: our model
    print("infer: ours")
    our_model.eval()
    our_out = our_model(our_x, pred_frames=pred_frames)[0]

    # checks
    print("check results")
    theirs = their_out.detach().cpu().numpy()
    ours = our_out.detach().cpu().numpy()
    if theirs.shape != ours.shape:
        raise AssertionError(f"Prediction shapes are not equal. "
                             f"Theirs: {theirs.shape}, ours: {ours.shape}")
    # save_arr_hist(np.abs(theirs - ours), test_id)
    if not np.allclose(theirs, ours, rtol=0, atol=1e-4):
        raise AssertionError("Predictions are not equal.")


def their_model_forward(their_model, x, context_frames, pred_frames):
    # taken from https://github.com/vincent-leguen/PhyDNet/blob/master/main.py
    with torch.no_grad():
        for t in range(context_frames - 1):
            encoder_output, encoder_hidden, _, _, _ = their_model(x[:, t], (t == 0))
        decoder_input = x[:, -1]  # first decoder input = last image of input sequence
        predictions = []
        for di in range(pred_frames):
            decoder_output, decoder_hidden, output_image, _, _ = their_model(decoder_input, False, False)
            decoder_input = output_image
            predictions.append(output_image)
        predictions = torch.stack(predictions, dim=1)
        return predictions
