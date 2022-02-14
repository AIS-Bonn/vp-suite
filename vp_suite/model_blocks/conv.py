r"""
This module contains convolutional model blocks.
"""

from torch import nn as nn

from vp_suite.base.base_model_block import ModelBlock


class DoubleConv2d(ModelBlock):
    r"""

    """
    NAME = "DoubleConv2d"

    def __init__(self, in_c, out_c):
        r"""

        Args:
            in_c ():
            out_c ():
        """
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        r"""

        Args:
            x ():

        Returns:

        """
        return self.conv(x)


class DoubleConv3d(ModelBlock):
    r"""

    """
    NAME = "DoubleConv3d"

    def __init__(self, in_c, out_c):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        r"""

        Args:
            x ():

        Returns:

        """
        return self.conv(x)


class DCGANConv(ModelBlock):
    r"""

    """
    NAME = "DCGAN - Conv"

    def __init__(self, nin, nout, stride):
        r"""

        Args:
            nin ():
            nout ():
            stride ():
        """
        super(DCGANConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        r"""

        Args:
            input ():

        Returns:

        """
        return self.main(input)


class DCGANConvTranspose(ModelBlock):
    r"""

    """
    NAME = "DCGAN - ConvTranspose"

    def __init__(self, nin, nout, stride):
        r"""

        Args:
            nin ():
            nout ():
            stride ():
        """
        super(DCGANConvTranspose, self).__init__()
        output_pad = int(stride == 2)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=(output_pad, output_pad)),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        r"""

        Args:
            input ():

        Returns:

        """
        return self.main(input)