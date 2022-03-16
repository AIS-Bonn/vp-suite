r"""
This module hosts several popular image-wise measures, such as Mean-Square Error (MSE),
Structural Similarity Index (SSIM) or Learned Perceptual Image Patch Similarity (LPIPS).

APPLIES TO ALL LOSSES:

- expected data type: torch.Tensor (data type: torch.float)

- expected shape: [b, t, c, h, w] ([b, t, 3, h, w] for LPIPS and SSIM)
"""

import torch
from torch import nn as nn
import piqa

from vp_suite.base import VPMeasure


class MSE(VPMeasure):
    r"""
    This class implements the pixel-wise Mean-Square Error (MSE/L2).
    """
    NAME = "Mean Squared Error (MSE) / L2 Loss"

    def __init__(self, device):
        super(MSE, self).__init__(device)
        self.criterion = nn.MSELoss(reduction="none").to(device)


class L1(VPMeasure):
    r"""
    This class implements the pixel-wise Mean Absolute Error (MAE/L1).
    """
    NAME = "Mean Absolute Error (MAE) / L1 Loss"

    def __init__(self, device):
        super(L1, self).__init__(device)
        self.criterion = nn.L1Loss(reduction="none").to(device)


class SmoothL1(VPMeasure):
    r"""
    This class implements a smoothed L1 Loss, resembling the MSE/L2 loss for smaller discrepancies and
    transitioning to the MAE/L1 loss for larger discrepancies.
    """
    NAME = "Smooth L1 Loss"
    
    def __init__(self, device):
        super(SmoothL1, self).__init__(device)
        self.criterion = nn.SmoothL1Loss(reduction="none").to(device)


class PSNR(VPMeasure):
    r"""
    This class implements the Peak Signal-to-Noise Ratio, which is related to the MSE.
    """
    NAME = "Peak Signal to Noise Ratio (PSNR)"
    BIGGER_IS_BETTER = True
    OPT_VALUE = float("inf")

    def __init__(self, device):
        super(PSNR, self).__init__(device)
        self.criterion = nn.MSELoss(reduction="none").to(device)

    def forward(self, pred, target):
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{self.NAME} expects 5-D inputs!")
        b, t, _, _, _ = pred.shape
        mses = self.criterion(pred, target).mean(dim=(-1, -2, -3))  # [b, t]
        psnr_losses = torch.log10(mses) * 10
        return psnr_losses.mean(dim=1).mean(dim=0)

    @classmethod
    def to_display(cls, x):
        return -x


class LPIPS(VPMeasure):
    r"""
    This class implements the "Learned Perceptual Image Patch Similarity (LPIPS)"
    from Zhang et al. (https://arxiv.org/abs/1801.03924), a perceptual measure that uses a pre-trained CNN to obtain
    features from the given prediction and ground truth.
    These perceptual features are then compared to yield the actual measurement value.
    """
    NAME = "Learned Perceptual Image Patch Similarity (LPIPS)"
    REFERENCE = "https://arxiv.org/abs/1801.03924"

    def __init__(self, device):
        super(LPIPS, self).__init__(device)
        self.criterion = piqa.lpips.LPIPS().to(device)

    def forward(self, pred, target):
        if pred.shape[2] != 3 or target.shape[2] != 3:
            raise ValueError(f"{self.NAME} needs 3-channel images with the channels at dim 2")
        pred, target = self.reshape_clamp(pred, target)
        return self.criterion(pred, target)


class SSIM(VPMeasure):
    r"""
    This class implements the structural similarity index (SSIM),
    as introduced in Zhou et al. (https://ieeexplore.ieee.org/document/1284395).
    """
    NAME = "Structural Similarity (SSIM)"
    REFERENCE = "https://ieeexplore.ieee.org/document/1284395"
    BIGGER_IS_BETTER = True
    OPT_VALUE = 1

    def __init__(self, device):
        super(SSIM, self).__init__(device)
        self.criterion = piqa.ssim.SSIM().to(device)

    def forward(self, pred, target):
        if pred.shape[2] != 3 or target.shape[2] != 3:
            raise ValueError(f"{self.NAME} needs 3-channel images with the channels at dim 2")
        pred, target = self.reshape_clamp(pred, target)
        return 1.0 - self.criterion(pred, target)

    @classmethod
    def to_display(cls, x):
        return 1.0 - x
