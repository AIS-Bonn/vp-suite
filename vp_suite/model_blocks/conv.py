r"""
This module contains convolutional model blocks.
"""
from torch import nn as nn

from vp_suite.base import VPModelBlock


class DoubleConv2d(VPModelBlock):
    r"""
    This class implements a 2D double-conv block, as used in the popular UNet architecture
    (Ronneberger et al., arxiv.org/abs/1505.04597).
    """
    NAME = "DoubleConv2d"
    PAPER_REFERENCE = "arxiv.org/abs/1505.04597"

    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv3d(VPModelBlock):
    r"""
    The class implements a 3D double-conv block, an extension of the :class:`DoubleConv2d` block
    to also process the time dimension.
    """
    NAME = "DoubleConv3d"

    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DCGANConv(VPModelBlock):
    r"""
    The class implements a DCGAN conv layer, as introduced in Radford et al. (arxiv.org/abs/1511.06434).
    """
    NAME = "DCGAN - Conv"
    PAPER_REFERENCE = "arxiv.org/abs/1511.06434"

    def __init__(self, in_channels, out_channels, stride):
        super(DCGANConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class DCGANConvTranspose(VPModelBlock):
    r"""
    The class implements a DCGAN convTranspose layer, as introduced in Radford et al. (arxiv.org/abs/1511.06434).
    """
    NAME = "DCGAN - ConvTranspose"
    PAPER_REFERENCE = "arxiv.org/abs/1511.06434"

    def __init__(self, in_channels, out_channels, stride):
        super(DCGANConvTranspose, self).__init__()
        output_pad = int(stride == 2)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=stride, padding=1, output_padding=(output_pad, output_pad)),
            nn.GroupNorm(16, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)
