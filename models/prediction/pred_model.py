import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


from models.model_blocks import DoubleConv2d, DoubleConv3d

class VideoPredictionModel(nn.Module):

    def __init__(self):
        super(VideoPredictionModel, self).__init__()

    def forward(self, x):
        # input: T frames: [b, T, c, h, w]
        # output: single frame: [b, c, h, w]
        raise NotImplementedError

    def pred_n(self, x, pred_length=1):
        # input: T frames: [b, T, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]
        preds = []
        for i in range(pred_length):
            pred = self.forward(x).unsqueeze(dim=1)
            preds.append(pred)
            x = torch.cat([x[:, 1:], pred], dim=1)
        return torch.cat(preds, dim=1)


class CopyLastFrameModel(VideoPredictionModel):

    def __init__(self):
        super(CopyLastFrameModel, self).__init__()

    def forward(self, x):
        return x[:, -1, :, :, :]


class UNet3d(VideoPredictionModel):

    def __init__(self, in_channels=3, out_channels=3, time_dim=4, features=[8, 16, 32, 64]):
        super(UNet3d, self).__init__()

        self.time_dim=time_dim
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


    def forward(self, x):
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
        return self.final_conv(x)


def test():
    import time

    batch_size = 8
    time_dim = 5
    num_channels = 3
    pred_length = 4
    img_size = 256, 256
    x = torch.randn((batch_size, time_dim, num_channels, *img_size))

    models = [
        CopyLastFrameModel(),
        UNet3d(in_channels=num_channels, out_channels=num_channels, time_dim=time_dim)
    ]

    for model in models:
        print("")
        print(f"Checking {model.__class__.__name__}")
        print(f"Parameter count (total / learnable): {sum([p.numel() for p in model.parameters()])}"
              f" / {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        t_start = time.time()
        pred1 = model(x)
        t_pred1 = round(time.time() - t_start, 6)

        t_start = time.time()
        preds = model.pred_n(x, pred_length)
        t_preds = round(time.time() - t_start, 6)

        print(f"Pred time (1 out frame / {pred_length} out frames): {t_pred1}s / {t_preds}s")
        print(f"Shapes ({time_dim} in frames / 1 out frame / {pred_length} out frames): "
              f"{list(x.shape)} / {list(pred1.shape)} / {list(preds.shape)}")

if __name__ == '__main__':
    test()