import torch
from torch import nn as nn
from torchvision.transforms import functional as TF

from models.model_blocks import DoubleConv3d, DoubleConv2d
from models.prediction.pred_model import VideoPredictionModel


class UNet3dModel(VideoPredictionModel):

    def __init__(self, in_channels=3, out_channels=3, time_dim=4, features=[8, 16, 32, 64]):
        super(UNet3dModel, self).__init__()

        self.time_dim = time_dim
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.time3ds = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        for feature in features:
            self.downs.append(DoubleConv3d(in_c=in_channels, out_c=feature))
            self.time3ds.append(nn.Conv3d(in_channels=feature, out_channels=feature, kernel_size=(time_dim, 1, 1)))
            in_channels = feature

        self.time3ds.append(nn.Conv3d(in_channels=features[-1], out_channels=features[-1], kernel_size=(time_dim, 1, 1)))

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv2d(in_c=feature * 2, out_c=feature))

        self.bottleneck = DoubleConv2d(in_c=features[-1], out_c=features[-1] * 2)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)


    def forward(self, x, **kwargs):
        # input: T frames: [b, T, c, h, w]
        # output: single frame: [b, c, h, w]
        assert x.shape[1] == self.time_dim, f"{self.time_dim} frames needed as pred input, {x.shape[1]} are given"
        x = x.permute((0, 2, 1, 3, 4))  # [b, c, T, h, w]
        skip_connections = []

        # DOWN
        for i in range(len(self.downs)):
            x = self.downs[i](x)

            skip_connection = self.time3ds[i](x).squeeze(dim=2)
            skip_connections.append(skip_connection)
            x = self.pool(x)

        x = self.time3ds[-1](x).squeeze(dim=2)  # from [b, feat[-1], T, h, w] to [b, feat[-1], h, w]
        x = self.bottleneck(x)


        # UP
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        # FINAL
        return self.final_conv(x), None