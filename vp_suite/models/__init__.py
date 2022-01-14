import sys
sys.path.append("")

from convlstm import ConvLSTM
from copy_last_frame import CopyLastFrame
from phydnet import PhyDNet
from st_lstm import STLSTM
from st_phy import STPhy
from unet3d import UNet3D
from lstm import LSTM
from simple import SimpleV1, SimpleV2

MODEL_CLASSES = {
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
AVAILABLE_MODELS = MODEL_CLASSES.keys()