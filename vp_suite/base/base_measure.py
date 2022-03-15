import torch
from torch import nn as nn


class VPMeasure(nn.Module):
    r"""
    The base class for all measures (nn Modules taking as input a ground truth sequence and a predicted sequence and
    providing a numerical assessment of the prediction quality). Measures can be losses and/or metrics, depending on
    their registration status in the base package's `__init__.py` file.
    All implemented losses and metrics should subclass this class.

    Attributes:
        device (str): A string specifying whether to use the GPU for calculations (`cuda`) or the CPU (`cpu`).

    Note:
        All measures that should be usable as losses should return values where lower means better.
        If for the specific measure higher means better, the actual value should be inverted in the forward method and
        inverted again in the :meth:`display()` method (which prepares the value for display to humans).
    """

    NAME: str = NotImplemented  #: The clear-text name of the measure.
    REFERENCE: str = None  #: The reference publication where this measure is originally introduced (represented as string)
    BIGGER_IS_BETTER = False  #: Specifies whether bigger values are better.
    OPT_VALUE = 0.  #: Specifies the best value attainable (e.g. when input tensors are equal).

    def __init__(self, device: str):
        r"""
        Instantiates the measure class by setting the device.
        Additionally, for the derived measure classes,
        instantiates the criterion that is used to calculate the measure value.

        Args:
            device (str): A string specifying whether to use the GPU for calculations (`cuda`) or the CPU (`cpu`).
        """
        super(VPMeasure, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        The module's forward pass takes the predicted frame sequence and the ground truth,
        compares them based on the deriving measure's criterion and logic and outputs a numerical assessment of the
        prediction quality.

        The base measure's forward method can be used by deriving classes and simply applies the criterion to the
        input tensors, sums up over all entries of an image and finally averages over frames and then batches.

        Args:
            pred (torch.Tensor): The predicted frame sequence as a 5D tensor (batch, frames, c, h, w).
            target (torch.Tensor): The ground truth frame sequence as a 5D tensor (batch, frames, c, h, w)

        Returns: The calculated numerical quality assessment.
        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{self.NAME} expects 5-D inputs!")
        value = self.criterion(pred, target)
        return value.sum(dim=(4, 3, 2)).mean(dim=1).mean(dim=0)

    def reshape_clamp(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Reshapes and clamps the input tensors, returning a 4D tensor where batch and time dimension are combined.

        Args:
            pred (torch.Tensor): The predicted frame sequence as a 5D tensor (batch, frames, c, h, w).
            target (torch.Tensor): The ground truth frame sequence as a 5D tensor (batch, frames, c, h, w)

        Returns: the reshaped and clamped pred and target tensors.
        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{self.NAME} expects 5-D inputs!")
        pred = pred.reshape(-1, *pred.shape[2:])  # [b*t, ...]
        pred = ((pred + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        target = target.reshape(-1, *target.shape[2:])  # [b*t, ...]
        target = ((target + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        return pred, target

    @classmethod
    def to_display(cls, x: float):
        r"""
        Converts a measurement value from the lower-is-better representation returned by the :meth:`forward()` method
        to the actual representation of the measure (e.g. SSIM having its best value at 1.0). If the measure did not
        get inverted in the :meth:`forward()`, this method just returns the input value.

        Args:
            x (float): The value to be converted.

        Returns: The converted value.
        """
        return x
