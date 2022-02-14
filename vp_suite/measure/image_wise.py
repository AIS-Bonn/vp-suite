r"""Module for image-wise measures.

APPLIES TO ALL LOSSES:
- expected data type: torch.tensor (torch.float)
- expected shape: [..., h, w] ([..., 3, h, w] for LPIPS and SSIM)
- expected value range: [-1.0, 1.0]
"""
import torch
from torch import nn as nn
import piqa

from vp_suite.base.base_measure import BaseMeasure


class MSE(BaseMeasure):
    r"""

    """
    NAME = "Mean Squared Error (MSE) / L2 Loss"
    def __init__(self, device):
        super(MSE, self).__init__(device)
        self.criterion = nn.MSELoss(reduction="none").to(device)

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        value = self.criterion(pred, real)
        return value.sum(dim=(-1, -2)).mean()


class L1(BaseMeasure):
    r"""

    """
    NAME = "Mean Absolute Error (MAE) / L1 Loss"
    def __init__(self, device):
        super(L1, self).__init__(device)
        self.criterion = nn.L1Loss(reduction="none").to(device)

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        value = self.criterion(pred, real)
        return value.sum(dim=(-1, -2)).mean()


class SmoothL1(BaseMeasure):
    r"""

    """
    NAME = "Smooth L1 Loss"
    def __init__(self, device):
        super(SmoothL1, self).__init__(device)
        self.criterion = nn.SmoothL1Loss(reduction="none").to(device)

    def forward(self, pred, real):
        value = self.criterion(pred, real)
        return value.sum(dim=(-1, -2)).mean()


class PSNR(BaseMeasure):
    r"""

    """
    NAME = "Peak Signal to Noise Ratio (PSNR)"
    BIGGER_IS_BETTER = True
    OPT_VALUE = float("inf")

    def __init__(self, device):
        super(PSNR, self).__init__(device)
        self.criterion = piqa.psnr.PSNR().to(device)

    def forward(self, pred, target):
        if pred.ndim == 5:
            pred = pred.reshape(-1, *pred.shape[2:])  # [b*t, ...]
        if target.ndim == 5:
            target = target.reshape(-1, *target.shape[2:])  # [b*t, ...]
        pred = ((pred + 1) / 2).clamp_(min=0.0, max=1.0) # range: [0., 1.]
        target = ((target + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        return -self.criterion(pred, target)

    @classmethod
    def to_display(cls, x):
        return -x


class LPIPS(BaseMeasure):
    r"""

    """
    NAME = "Learned Perceptual Image Patch Similarity (LPIPS)"
    REFERENCE = "https://arxiv.org/abs/1801.03924"

    def __init__(self, device):
        super(LPIPS, self).__init__(device)
        self.criterion = piqa.lpips.LPIPS().to(device)

    def forward(self, pred, target):
        if pred.shape[-3] != 3 or target.shape[-3] != 3:
            raise ValueError("LPIPS needs 3-channel images with the channels at dim -3")
        pred = pred.reshape(-1, *pred.shape[-3:])  # [..., 3, h, w]
        target = target.reshape(-1, *target.shape[-3:])  # [..., 3, h, w]
        pred = ((pred + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        target = ((target + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        return self.criterion(pred, target)  # scalar


class SSIM(BaseMeasure):
    r"""

    """
    NAME = "Structural Similarity (SSIM)"
    REFERENCE = "https://ieeexplore.ieee.org/document/1284395"
    BIGGER_IS_BETTER = True
    OPT_VALUE = 1

    def __init__(self, device):
        super(SSIM, self).__init__(device)
        self.criterion = piqa.ssim.SSIM().to(device)

    def forward(self, pred, target):
        if pred.shape[-3] != 3 or target.shape[-3] != 3:
            raise ValueError("LPIPS needs 3-channel images with the channels at dim -3")
        pred = pred.reshape(-1, *pred.shape[-3:])  # [..., 3, h, w]
        target = target.reshape(-1, *target.shape[-3:])  # [..., 3, h, w]
        pred = ((pred + 1) / 2).clamp_(min=0.0, max=1.0) # range: [0., 1.]
        target = ((target + 1) / 2).clamp_(min=0.0, max=1.0)  # range: [0., 1.]
        return 1.0 - self.criterion(pred, target)

    @classmethod
    def to_display(cls, x):
        return 1.0 - x
