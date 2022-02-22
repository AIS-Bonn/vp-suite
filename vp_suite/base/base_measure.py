import torch
from torch import nn as nn


class BaseMeasure(nn.Module):
    r"""

    """
    NAME: str = NotImplemented  #: The clear-text name of the measure.
    REFERENCE: str = None  #: The reference where this measure is originally introduced.
    BIGGER_IS_BETTER = False  #: specifying whether bigger values are better
    OPT_VALUE = 0.  #: specifying the best value attainable (e.g. with equal tensors)

    def __init__(self, device):
        r"""

        Args:
            device ():
        """
        super(BaseMeasure, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""

        Args:
            pred ():
            target ():

        Returns:

        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{self.NAME} expects 5-D inputs!")
        value = self.criterion(pred, target)
        return value.sum(dim=(4, 3, 2)).mean(dim=1).mean(dim=0)

    def reshape_clamp(self, pred: torch.Tensor, target: torch.Tensor):
        r"""

        Args:
            pred ():
            target ():

        Returns:

        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{self.NAME} expects 5-D inputs!")
        pred = pred.reshape(-1, *pred.shape[2:])  # [b*t, ...]
        pred = ((pred + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        target = target.reshape(-1, *target.shape[2:])  # [b*t, ...]
        target = ((target + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        return pred, target

    @classmethod
    def to_display(cls, x):
        r"""

        Args:
            x ():

        Returns:

        """
        return x
