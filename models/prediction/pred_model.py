import sys
sys.path.append(".")

import torch
import torch.nn as nn

from config import DEVICE
from models.prediction.conv_lstm import LSTMModel
from models.prediction.copy_last_frame import CopyLastFrameModel
from models.prediction.st_lstm import STLSTMModel
from models.prediction.unet_3d import UNet3dModel


class VideoPredictionModel(nn.Module):

    def __init__(self):
        super(VideoPredictionModel, self).__init__()

    def forward(self, x, **kwargs):
        # input: T frames: [b, T, c, h, w]
        # output: single frame: [b, c, h, w]
        raise NotImplementedError

    def pred_n(self, x, pred_length=1, **kwargs):
        # input: T frames: [b, T, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]
        preds = []
        loss_dicts = []
        for i in range(pred_length):
            pred, loss_dict = self.forward(x)
            pred = pred.unsqueeze(dim=1)
            preds.append(pred)
            loss_dicts.append(loss_dict)
            x = torch.cat([x[:, 1:], pred], dim=1)

        pred = torch.cat(preds, dim=1)
        if loss_dicts[0] is not None:
            loss_dict = {k: torch.mean([loss_dict[k] for loss_dict in loss_dicts]) for k in loss_dicts[0]}
        else:
            loss_dict = None
        return pred, loss_dict


def test():

    import time

    batch_size = 6
    time_dim = 5
    num_channels = 23
    pred_length = 5
    img_size = 135, 240
    action_size = 3
    x = torch.randn((batch_size, time_dim, num_channels, *img_size)).to(DEVICE)
    a = torch.randn((batch_size, time_dim+pred_length, action_size)).to(DEVICE)

    models = [
        CopyLastFrameModel(),
        UNet3dModel(in_channels=num_channels, out_channels=num_channels, time_dim=time_dim).to(DEVICE),
        LSTMModel(in_channels=num_channels, out_channels=num_channels).to(DEVICE),
        STLSTMModel(img_size, img_channels=num_channels, device=DEVICE),
        STLSTMModel(img_size, img_channels=num_channels, device=DEVICE, action_size=action_size),
    ]

    for model in models:
        print("")
        print(f"Checking {model.__class__.__name__}")
        print(f"Parameter count (total / learnable): {sum([p.numel() for p in model.parameters()])}"
              f" / {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        t_start = time.time()
        pred1, _ = model(x, actions=a)
        t_pred1 = round(time.time() - t_start, 6)

        t_start = time.time()
        preds, _ = model.pred_n(x, pred_length, actions=a)
        t_preds = round(time.time() - t_start, 6)

        print(f"Pred time (1 out frame / {pred_length} out frames): {t_pred1}s / {t_preds}s")
        print(f"Shapes ({time_dim} in frames / 1 out frame / {pred_length} out frames): "
              f"{list(x.shape)} / {list(pred1.shape)} / {list(preds.shape)}")


if __name__ == '__main__':
    test()