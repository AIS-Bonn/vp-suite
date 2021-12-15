import sys
sys.path.append(".")

import torch
import numpy as np

from metrics.segmentation import Accuracy as ACC
from losses.loss_image import LPIPS, MSE, L1, SmoothL1, SSIM, PSNR


def get_segmentation_metrics(pred, target):
    '''
    input type: torch.tensor (torch.int)
    input shape: [b, h, w]
    input range: class labels starting from 0
    '''

    pred_stacked = list(pred.detach().cpu())
    target_stacked = list(target.detach().cpu())

    return {
        "accuracy (↑)": np.mean([ACC(p, t) for p, t in zip(pred_stacked, target_stacked)])
    }


class PredictionMetricProvider():
    def __init__(self, cfg):

        self.device = cfg.device
        self.metrics = {
            "mse": MSE(device=self.device),
            "mae/l1": L1(device=self.device),
            "smooth_l1": SmoothL1(device=self.device),
            "lpips": LPIPS(device=self.device),
            "ssim": SSIM(device=self.device),
            "psnr": PSNR(device=self.device)
        }
        # FVD loss only available for 2- or 3- channel input
        self.use_fvd = cfg.num_channels in [2, 3]

    def get_metrics(self, pred, target, frames=None, all_frame_cnts=False):
        '''
        input type: torch.tensor (torch.float)
        input shape: [b, t, c, h, w]
        input range: [-1.0, 1.0]
        if frames is specified, only considers the first 'frames' frames.
        '''

        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")

        frames = frames or pred.shape[1]
        pred = pred[:, :frames]
        target = target[:, :frames]

        metrics_dict = {}
        frames = [frames] if not all_frame_cnts else range(1, frames + 1)

        for frame_cnt in frames:
            pred_ = pred[:, :frame_cnt]
            target_ = target[:, :frame_cnt]
            frame_cnt_metrics = {
                f"{key}_{frame_cnt} ({'↑' if loss.bigger_is_better else '↓'})":
                    loss.loss_to_display(loss(pred_, target_).item())
                for key, loss in self.metrics.items()
            }
            metrics_dict.update(frame_cnt_metrics)

        return metrics_dict