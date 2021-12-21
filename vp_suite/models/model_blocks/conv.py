from torch import nn as nn


class DoubleConv2d(nn.Module):
    def __init__(self, in_c, out_c):
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
        return self.conv(x)


class DoubleConv3d(nn.Module):
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
        return self.conv(x)


class DCGANConv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(DCGANConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class DCGANConvTranspose(nn.Module):
    def __init__(self, nin, nout, stride):
        super(DCGANConvTranspose, self).__init__()
        output_pad = int(stride == 2)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=(output_pad, output_pad)),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)