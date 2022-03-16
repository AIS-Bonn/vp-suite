r"""
This modules contains all usable video prediction models.
"""

from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.models.phydnet import PhyDNet
from vp_suite.models.st_phy import STPhy
from vp_suite.models.unet3d import UNet3D
from vp_suite.models.lstm import LSTM
from vp_suite.models.predrnn_v2 import PredRNN_V2
from vp_suite.models.precipitation_nowcasting.ef_conv_lstm import EF_ConvLSTM
from vp_suite.models.precipitation_nowcasting.ef_traj_gru import EF_TrajGRU

MODEL_CLASSES = {
    # simple/baseline models
    "copy": CopyLastFrame,
    "lstm": LSTM,
    "unet-3d": UNet3D,

    # more sophisticated models
    "phy": PhyDNet,
    "st-phy": STPhy,
    "convlstm-shi": EF_ConvLSTM,
    "trajgru": EF_TrajGRU,
    "predrnn-pp": PredRNN_V2,
}  #: A dictionary of all models and the corresponding string identifiers with which they can be accessed.

AVAILABLE_MODELS = MODEL_CLASSES.keys()
