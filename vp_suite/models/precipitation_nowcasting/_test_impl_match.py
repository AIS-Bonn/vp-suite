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
from vp_suite.models.precipitation_nowcasting.ef_traj_gru import EF_TrajGRU as OurEF_TrajGRU
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
    model_args = {"img_shape": (c, h, w), "action_size": 0, "tensor_value_range": [0.0, 1.0]}
    our_convlstm_EF = OurEF_ConvLSTM(device, **model_args).to(device)
    our_trajgru_EF = OurEF_TrajGRU(device, **model_args).to(device)

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
    #
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


if __name__ == '__main__':
    compare_implementations()