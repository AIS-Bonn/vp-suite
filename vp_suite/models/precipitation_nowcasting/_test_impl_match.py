from copy import deepcopy
import sys
from unittest.mock import Mock

sys.path.append("./Precipitation-Nowcasting")

import torch
import numpy as np

sys.modules["pandas"] = Mock()
sys.modules["nowcasting.hko.evaluation"] = Mock()
sys.modules["nowcasting.train_and_test"] = Mock()

from nowcasting.config import cfg  # needs to be imported due to circular import in their code
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF as TheirEF
from experiments.net_params import encoder_params, forecaster_params, \
    convlstm_encoder_params, convlstm_forecaster_params
from vp_suite.models.precipitation_nowcasting.ef_traj_gru import EF_TrajGRU as OurEF_TrajGRU, Activation
from vp_suite.models.precipitation_nowcasting.ef_conv_lstm import EF_ConvLSTM as OurEF_ConvLSTM
from vp_suite.utils.models import state_dicts_equal


def compare_implementations():
    r"""
    IMPORTANT:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    context_frames, pred_frames = 5, 20  # default values for their EF model
    c, h, w = 1, 480, 480  # their default EF model takes in images of shape (1, 480, 480)

    # set up original models
    print("setting up their models")
    their_convlstm_encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(device)
    their_convlstm_forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(device)
    their_convlstm_EF = TheirEF(their_convlstm_encoder, their_convlstm_forecaster).to(device)

    their_trajgru_encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    their_trajgru_forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(device)
    their_trajgru_EF = TheirEF(their_trajgru_encoder, their_trajgru_forecaster).to(device)

    # set up our models
    print("setting up our models")
    our_convlstm_EF = OurEF_ConvLSTM(device, **get_HKO_config_ConvLSTM(c, h, w)).to(device)
    our_trajgru_EF = OurEF_TrajGRU(device, **get_HKO_config_TrajGRU(c, h, w)).to(device)

    for (their_model, our_model) in [(their_trajgru_EF, our_trajgru_EF), (their_convlstm_EF, our_convlstm_EF)]:
        print(f"CHECK: {our_model.NAME}")

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


def get_HKO_config_TrajGRU(c, h, w):
    return {
        "img_shape": (c, h, w),
        "action_size": 0,
        "tensor_value_range": [0.0, 1.0],

        "activation": Activation('leaky', negative_slope=0.2, inplace=True),
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

        "enc_rnn_z": [0.0, 0.0, 0.0],
        "enc_rnn_L": [13, 13, 9],
        "enc_rnn_i2h_k": [(3, 3), (3, 3), (3, 3)],
        "enc_rnn_i2h_s": [(1, 1), (1, 1), (1, 1)],
        "enc_rnn_i2h_p": [(1, 1), (1, 1), (1, 1)],
        "enc_rnn_h2h_k": [(5, 5), (5, 5), (3, 3)],
        "enc_rnn_h2h_d": [(1, 1), (1, 1), (1, 1)],

        "dec_rnn_z": [0.0, 0.0, 0.0],
        "dec_rnn_L": [13, 13, 9],
        "dec_rnn_i2h_k": [(3, 3), (3, 3), (3, 3)],
        "dec_rnn_i2h_s": [(1, 1), (1, 1), (1, 1)],
        "dec_rnn_i2h_p": [(1, 1), (1, 1), (1, 1)],
        "dec_rnn_h2h_k": [(3, 3), (5, 5), (5, 5)],
        "dec_rnn_h2h_d": [(1, 1), (1, 1), (1, 1)],

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


if __name__ == '__main__':
    compare_implementations()