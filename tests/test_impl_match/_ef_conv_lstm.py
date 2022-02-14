from copy import deepcopy
import sys, os

import torch
import numpy as np

from vp_suite.models import EF_ConvLSTM as OurEF_ConvLSTM
from vp_suite.utils.models import state_dicts_equal

REFERENCE_GIT_URL = "https://github.com/Hzzone/Precipitation-Nowcasting.git"
REPO_DIR = "Precipitation-Nowcasting"


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

    from nowcasting.config import cfg
    from unittest.mock import Mock
    sys.modules["pandas"] = Mock()
    sys.modules["nowcasting.hko.evaluation"] = Mock()
    sys.modules["nowcasting.train_and_test"] = Mock()
    from nowcasting.models.forecaster import Forecaster
    from nowcasting.models.encoder import Encoder
    from nowcasting.models.model import EF as TheirEF
    from experiments.net_params import encoder_params, forecaster_params, \
        convlstm_encoder_params, convlstm_forecaster_params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    context_frames, pred_frames = 5, 20  # default values for their EF model
    c, h, w = 1, 480, 480  # their default EF model takes in images of shape (1, 480, 480)

    # set up original models
    print("setting up their model")
    their_convlstm_encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(device)
    their_convlstm_forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(device)
    their_model = TheirEF(their_convlstm_encoder, their_convlstm_forecaster).to(device)

    # set up our models
    print("setting up our model")
    our_model = OurEF_ConvLSTM(device, **get_HKO_config_ConvLSTM(c, h, w)).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    if not state_dicts_equal(their_model, our_model):
        raise AssertionError("State dicts not equal!")
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    if not state_dicts_equal(their_model, our_model, check_values=True):
        raise AssertionError("State dicts not equal!")

    # set up input
    print("setting up input")
    their_x = torch.rand(context_frames, batch_size, c, h, w, device=device)
    our_x = their_x.clone().permute((1, 0, 2, 3, 4))

    # infer: their model
    print("infer: theirs")
    their_model.eval()
    their_out = their_model(their_x)
    print(their_out.shape)

    # infer: our model
    print("infer: ours")
    our_model.eval()
    our_out = our_model(our_x, pred_frames=pred_frames)[0].permute((1, 0, 2, 3, 4))

    # checks
    print("check results")
    theirs = their_out.detach().cpu().numpy()
    ours = our_out.detach().cpu().numpy()
    if theirs.shape != ours.shape:
        raise AssertionError(f"Prediction shapes are not equal. "
                             f"Theirs: {theirs.shape}, ours: {ours.shape}")
    # save_diff_hist(np.abs(theirs - ours), test_id)
    if not np.allclose(theirs, ours, rtol=0, atol=1e-4):
        raise AssertionError("Predictions are not equal.")


def get_HKO_config_ConvLSTM(c, h, w):
    return {
        "img_shape": (c, h, w),
        "action_size": 0,
        "tensor_value_range": [0.0, 1.0],

        "num_layers": 3,
        "enc_c": [8, 64, 192, 192, 192, 192],
        "dec_c": [192, 192, 192, 64, 64, 8],
    
        "enc_conv_names": ["conv1_leaky_1", "conv2_leaky_1", "conv3_leaky_1"],
        "enc_conv_k": [7, 5, 3],
        "enc_conv_s": [5, 3, 2],
        "enc_conv_p": [1, 1, 1],
    
        "dec_conv_names": ["deconv1_leaky_1", "deconv2_leaky_1", "deconv3_leaky_1"],
        "dec_conv_k": [4, 5, 7],
        "dec_conv_s": [2, 3, 5],
        "dec_conv_p": [1, 1, 1],
    
        "enc_rnn_k": [3, 3, 3],
        "enc_rnn_s": [1, 1, 1],
        "enc_rnn_p": [1, 1, 1],
    
        "dec_rnn_k": [3, 3, 3],
        "dec_rnn_s": [1, 1, 1],
        "dec_rnn_p": [1, 1, 1],
    
        "final_conv_1_name": "conv3_leaky_2",
        "final_conv_1_c": 8,
        "final_conv_1_k": 3,
        "final_conv_1_s": 1,
        "final_conv_1_p": 1,
    
        "final_conv_2_name": "conv3_3",
        "final_conv_2_k": 1,
        "final_conv_2_s": 1,
        "final_conv_2_p": 0,
    }
