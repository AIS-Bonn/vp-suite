import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as TF
from torchvision.transforms import Resize

from vp_suite.models.model_blocks.conv import DCGANConv, DCGANConvTranspose


class Autoencoder(nn.Module):
    def __init__(self, img_shape, encoded_channels, device):
        super(Autoencoder, self).__init__()

        self.img_shape = img_shape
        self.img_h, self.img_w, self.img_c = img_shape
        self.enc_c = encoded_channels
        self.device = device

        self.build_models()
        self.to(self.device)

        zeros = torch.zeros((1, self.img_c, self.img_h, self.img_w), device=self.device)
        encoded_zeros = self.encoder(zeros)
        self.encoded_shape = encoded_zeros.shape
        self.encoded_numel = encoded_zeros.numel()

    def build_models(self):
        self.encoder = Encoder(in_channels=self.img_c, out_channels=self.enc_c)
        self.decoder = Decoder(in_channels=self.enc_c, out_shape=self.img_shape)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.act_fn = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.mean_layer = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, stride=1)

    def forward(self, x):

        x = self.act_fn(self.conv1(x))
        x = self.act_fn(self.conv2(x))
        x = self.act_fn(self.mean_layer(x))
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, out_shape):
        super().__init__()

        self.act_fn = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_h, self.out_w, self.out_c = out_shape

        self.fc1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1)
        self.conv1 = nn.ConvTranspose2d(self.in_channels, 64, kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(32, self.out_c, kernel_size=5, stride=1, padding=0)
        self.res = TF.Resize(size=(self.out_h, self.out_w))


    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.conv1(x))
        x = self.act_fn(self.conv2(x))
        x = self.res(self.conv3(x))
        return x


class DCGANEncoder(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(DCGANEncoder, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = DCGANConv(nc, nf, stride=2)  # (32) x 32 x 32
        self.c2 = DCGANConv(nf, nf, stride=1)  # (32) x 32 x 32
        self.c3 = DCGANConv(nf, 2 * nf, stride=2)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class DCGANDecoder(nn.Module):
    def __init__(self, out_size, nc=1, nf=32):
        super(DCGANDecoder, self).__init__()
        self.upc1 = DCGANConvTranspose(2 * nf, nf, stride=2)  # (32) x 32 x 32
        self.upc2 = DCGANConvTranspose(nf, nf, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=2, padding=1,
                                       output_padding=1)  # (nc) x 64 x 64
        self.resize = Resize(size=out_size)

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        out = self.resize(d3)
        return out