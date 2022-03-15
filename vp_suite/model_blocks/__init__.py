r"""
This package contains model blocks that can be used by the video prediction models.
"""
import sys
import inspect

from vp_suite.base import VPModelBlock

from .conv import DoubleConv2d, DoubleConv3d, DCGANConv, DCGANConvTranspose
from .enc import Autoencoder, Encoder, Decoder, DCGANEncoder, DCGANDecoder
from .phydnet import PhyCell, PhyCell_Cell
from .predrnn import SpatioTemporalLSTMCell, ActionConditionalSpatioTemporalLSTMCell
from .traj_gru import TrajGRU
from .conv_lstm_ndrplz import ConvLSTM as ConvLSTM_ndrplz
from .conv_lstm_hzzone import ConvLSTM


def is_model_block(obj):
    cls = obj if inspect.isclass(obj) else type(obj)
    return issubclass(cls, VPModelBlock) and not cls == VPModelBlock


MODEL_BLOCK_CLASSES = [cls for (cls_name, cls) in inspect.getmembers(sys.modules[__name__], is_model_block)]
