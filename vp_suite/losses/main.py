import torch

from vp_suite.losses.loss_image import MSE, L1, SmoothL1, LPIPS, SSIM, PSNR


class PredictionLossProvider():
    def __init__(self, cfg):

        self.ignore_zero_scales = not cfg.calc_zero_loss_scales
        self.device = cfg.device
        self.losses = {
            "mse": (MSE(device=self.device), cfg.mse_loss_scale),
            "l1": (L1(device=self.device), cfg.l1_loss_scale),
            "smooth_l1": (SmoothL1(device=self.device), cfg.smoothl1_loss_scale),
            "lpips": (LPIPS(device=self.device), cfg.lpips_loss_scale),
            "ssim": (SSIM(device=self.device), cfg.ssim_loss_scale),
            "psnr": (PSNR(device=self.device), cfg.psnr_loss_scale)
        }
        '''
        # FVD loss only available for 2- or 3- channel input
        if cfg.num_channels in [2, 3]:
            self.losses["fvd"] = (FVD(device=self.device, num_frames=cfg.vid_pred_length,
                                      in_channels=cfg.num_channels),
                                  cfg.fvd_loss_scale)
        '''

    def get_losses(self, pred, target, eval=False):
        '''
        input type: torch.tensor (torch.float)
        input shape: [b, t, c, h, w]
        input range: [-1.0, 1.0]
        '''
        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")

        loss_values, total_loss = {}, torch.tensor(0.0, device=self.device)
        for key, (loss, scale) in self.losses.items():
            if scale == 0 and self.ignore_zero_scales and not eval: continue
            val = loss(pred, target)
            total_loss += scale * val
            loss_values[key] = loss.loss_to_display(val)

        return loss_values, total_loss