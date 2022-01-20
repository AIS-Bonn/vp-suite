r"""
This package contains the measures.
Measures can either be losses or metrics depending on whether they're differentiable.
"""

from vp_suite.measure.image_wise import MSE, L1, SmoothL1, LPIPS, SSIM, PSNR
from vp_suite.measure.fvd.fvd import FrechetVideoDistance

# === losses ===================================================================

LOSS_CLASSES = {
    "mse": MSE,
    "l1": L1,
    "smooth_l1": SmoothL1,
    "lpips": LPIPS,
    "ssim": SSIM,
    "psnr": PSNR,
    "fvd": FrechetVideoDistance
}  #: TODO
AVAILABLE_LOSSES = LOSS_CLASSES.keys()

# === metrics ==================================================================

METRIC_CLASSES = {
    "mse": MSE,
    "l1": L1,
    "smooth_l1": SmoothL1,
    "lpips": LPIPS,
    "ssim": SSIM,
    "psnr": PSNR,
    "fvd": FrechetVideoDistance
}  #: TODO
AVAILABLE_METRICS = METRIC_CLASSES.keys()
