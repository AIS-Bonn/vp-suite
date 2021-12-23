from torch import nn as nn


class BaseMeasure(nn.Module):

    bigger_is_better = False

    def __init__(self, device):
        super(BaseMeasure, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, pred, target):
        pass

    @classmethod
    def to_display(cls, x): return x