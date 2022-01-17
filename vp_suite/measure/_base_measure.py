from torch import nn as nn


class BaseMeasure(nn.Module):

    BIGGER_IS_BETTER = False  # specifying whether bigger values are better
    OPT_VALUE = 0.  # specifying the best value attainable (e.g. with equal tensors)

    def __init__(self, device):
        super(BaseMeasure, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, pred, target):
        pass

    @classmethod
    def to_display(cls, x): return x
