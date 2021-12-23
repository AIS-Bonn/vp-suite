import torch

from vp_suite.measure.image_wise import MSE, L1, SmoothL1, LPIPS, SSIM, PSNR
from vp_suite.measure.fvd.fvd import FrechetVideoDistance as FVD

LOSSES = {
    "mse": MSE,
    "l1": L1,
    "smooth_l1": SmoothL1,
    "lpips": LPIPS,
    "ssim": SSIM,
    "psnr": PSNR,
    "fvd": FVD
}

AVAILABLE_LOSSES = LOSSES.keys()

class PredictionLossProvider():
    def __init__(self, cfg):

        self.device = cfg.device
        loss_scales = cfg.losses_and_scales
        if "fvd" in loss_scales.keys() and cfg.img_c not in [2, 3]:
            print("WARNING: 'FVD' measure won't be used since image channels needs to be in [2, 3]")
            loss_scales.pop("fvd")
        self.losses = {k: (LOSSES[k](device=self.device), scale) for k, scale in loss_scales.items()}

    def get_losses(self, pred, target):
        '''
        input type: torch.tensor (torch.float)
        input shape: [b, t, c, h, w]
        '''
        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")

        loss_display_values, total_loss = {}, torch.tensor(0.0, device=self.device)
        for key, (loss, scale) in self.losses.items():
            val = loss(pred, target)
            total_loss += scale * val
            loss_display_values[key] = loss.to_display(val)

        return loss_display_values, total_loss