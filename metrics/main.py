import sys
sys.path.append(".")

import torch
import numpy as np

from metrics.image_metrics import SSIM, PSNR, LPIPS, MSE, MAE
from metrics.fvd import FrechetVideoDistance as FVD
from config import DEVICE

def get_image_metrics(pred, target):
    '''
    input type: torch.tensor (torch.float)
    input shape: [b, t, c, h, w]
    input range: [-1.0, 1.0]
    '''

    # create zipper list of pred-target numpy array pairs, removing other dimensions
    pred = pred.view(-1, *pred.shape[2:]).detach().cpu()
    pred_torch, pred_numpy = list(pred), list(pred.numpy())
    target = target.view(-1, *target.shape[2:]).detach().cpu()
    target_torch, target_numpy = list(target), list(target.numpy())

    return {
        "ssim": np.mean(np.stack([SSIM(p, t) for p, t in zip(pred_numpy, target_numpy)]), axis=0),
        "psnr": np.mean(np.stack([PSNR(p, t) for p, t in zip(pred_numpy, target_numpy)]), axis=0),
        "mse": np.mean(np.stack([MSE(p, t) for p, t in zip(pred_numpy, target_numpy)]), axis=0),
        "mae": np.mean(np.stack([MAE(p, t) for p, t in zip(pred_numpy, target_numpy)]), axis=0),
        "lpips": torch.mean(torch.stack([LPIPS(p, t) for p, t in zip(pred_torch, target_torch)]), axis=0).item(),
    }

def get_video_metrics(pred, target):
    '''
    input type: torch.tensor (torch.float)
    input shape: [b, t, c, h, w]
    input range: [-1.0, 1.0]
    '''

    _, t, c, _, _ = pred.shape

    return {
        "fvd": FVD(num_frames=t, in_channels=c, device=pred.device).get_distance(pred, target).item()
    }


def get_metrics_for_video(pred, target):
    '''
    input type: torch.tensor (torch.float)
    input shape: [b, t, c, h, w]
    input range: [-1.0, 1.0]
    '''

    if pred.shape != target.shape:
        raise ValueError("Output images and target images are of different shape!")

    image_metrics = get_image_metrics(pred, target)
    video_metrics = get_video_metrics(pred, target)

    return {**image_metrics, **video_metrics}


if __name__ == '__main__':

    a, b = torch.rand(2, 50, 3, 93, 124).to(DEVICE), torch.rand(2, 50, 3, 93, 124).to(DEVICE)  # [b, t, c, h, w]
    a, b = 2*a-1, 2*b-1  # range: [-1.0, 1.0)

    for metric, val in get_metrics_for_video(a, b).items():
        print(f"{metric}: {val}")