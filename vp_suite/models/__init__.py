r"""
This modules contains all usable models.
"""

from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.models.phydnet import PhyDNet
from vp_suite.models.st_phy import STPhy
from vp_suite.models.unet3d import UNet3D
from vp_suite.models.lstm import LSTM
from vp_suite.models.simple import SimpleV1, SimpleV2
from vp_suite.models.predrnn.predrnn_v2 import PredRNN_V2

MODEL_CLASSES = {
    # simple/baseline models
    "copy": CopyLastFrame,
    "simple-v1": SimpleV1,
    "simple-v2": SimpleV2,
    "lstm": LSTM,
    "unet-3d": UNet3D,

    # more sophisticated models
    "phy": PhyDNet,
    "st-phy": STPhy,
    "predrnn-pp": PredRNN_V2,
}  #: A dictionary of all models

AVAILABLE_MODELS = MODEL_CLASSES.keys()
