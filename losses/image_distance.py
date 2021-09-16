import torch
from torch import nn as nn


# APPLIES TO ALL LOSSES:
# expected data type: torch.tensor (torch.float)
# expected shape: [..., c, h, w]
# expected value range: [-1.0, 1.0]

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        value = self.loss(pred, real)
        return value.sum(dim=(-1, -2)).mean()


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        value = self.loss(pred, real)
        return value.sum(dim=(-1, -2)).mean()


class SmoothL1(nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction="none")
        
    def forward(self, pred, real):
        value = self.loss(pred, real)
        return value.sum(dim=(-1, -2)).mean()