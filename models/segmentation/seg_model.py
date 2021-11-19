import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# inspired by https://www.youtube.com/watch?v=IHq1t7NxS8k
from models.model_blocks.conv import DoubleConv2d


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, features):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv2d(in_c=in_channels, out_c=feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv2d(in_c=feature * 2, out_c=feature))

        self.bottleneck = DoubleConv2d(in_c=features[-1], out_c=features[-1] * 2)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        # DOWN
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # BOTTLENECK
        x = self.bottleneck(x)

        # UP
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        # FINAL
        return self.final_conv(x)


def test():
    x = torch.randn((8, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=21, features=[64, 128, 256, 512])
    preds = model(x)
    print(x.shape, preds.shape)

if __name__ == '__main__':
    test()