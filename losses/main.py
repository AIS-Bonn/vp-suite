import sys
sys.path.append(".")

import torch
import torch.nn as nn
import numpy as np

from losses.image_distance import MSE, L1, SmoothL1
from losses.fvd import FrechetVideoDistance as FVD


class PredictionLossProvider(nn.Module):
    def __init__(self, num_channels, num_pred_frames, device, loss_scales:dict={}, ignore_zero_scales=True):

        self.ignore_zero_scales = ignore_zero_scales
        self.device = device
        self.losses = {
            "mse": (MSE().to(device), loss_scales.get("mse", 1.0)),
            "l1": (L1().to(device), loss_scales.get("l1", 1.0)),
            "smooth_l1": (SmoothL1().to(device), loss_scales.get("smooth_l1", 0.0))
        }
        # FVD loss only available for 2- or 3- channel input
        if num_channels in [2, 3]:
            self.losses["fvd"] = (FVD(num_frames=num_pred_frames, in_channels=num_channels).to(device),
                                  loss_scales.get("fvd", 0.0))

    def get_losses(self, pred, target, eval=False):
        '''
        input type: torch.tensor (torch.float)
        input shape: [b, t, c, h, w]
        input range: [-1.0, 1.0]
        '''

        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")
        b, t, c, h, w = pred.shape

        loss_values, total_loss = {}, torch.tensor(0.0, device=self.device)
        for key, (loss, scale) in self.losses.items():
            if scale == 0 and self.ignore_zero_scales and not eval: continue
            val = loss(pred, target)
            loss_values[key] = val
            total_loss += val * scale

        return loss_values, total_loss


if __name__ == '__main__':
    from run import DEVICE

    print("\nPrediction losses (after item() call):")
    a, b = torch.rand(8, 16, 3, 93, 124).to(DEVICE), torch.rand(8, 16, 3, 93, 124).to(DEVICE)  # [b, t, c, h, w]
    a, b = 2*a-1, 2*b-1  # range: [-1.0, 1.0)

    loss_provider = PredictionLossProvider(num_channels=3, num_pred_frames=16, device=DEVICE)
    loss_values, total_loss = loss_provider.get_losses(a, b)
    for metric, val in loss_values.items():
        print(f"{metric}: {val.item()}")
    print(f"TOTAL LOSS (incl. scale): {total_loss.item()}")
