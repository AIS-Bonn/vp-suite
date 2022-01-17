from vp_suite.models.convlstm import ConvLSTM
from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.models.phydnet import PhyDNet
from vp_suite.models.st_lstm import STLSTM
from vp_suite.models.st_phy import STPhy
from vp_suite.models.unet3d import UNet3D
from vp_suite.models.lstm import LSTM
from vp_suite.models.simple import SimpleV1, SimpleV2

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