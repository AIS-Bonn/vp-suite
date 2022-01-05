import sys
sys.path.append("")

import torch

from vp_suite.models.convlstm import ConvLSTM
from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.models.phydnet import PhyDNet
from vp_suite.models.st_lstm import STLSTM
from vp_suite.models.st_phy import STPhy
from vp_suite.models.unet3d import UNet3D
from vp_suite.models.lstm import LSTM
from vp_suite.models.simple import SimpleV1, SimpleV2

pred_models = {
    "unet": UNet3D,
    "lstm" : ConvLSTM,
    "non_conv" : LSTM,
    "st_lstm" : STLSTM,
    "copy" : CopyLastFrame,
    "phy" : PhyDNet,
    "st_phy" : STPhy,
    "simplev1": SimpleV1,
    "simplev2": SimpleV2,
}

AVAILABLE_MODELS = pred_models.keys()

def create_pred_model(trainer_config, model_type, **model_args):
    model_class = pred_models.get(model_type, pred_models["copy"])
    pred_model = model_class(trainer_config, **model_args).to(trainer_config["device"])
    return pred_model.to(trainer_config["device"])
