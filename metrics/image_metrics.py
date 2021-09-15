import numpy as np
import torch

import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

# APPLIES TO ALL METRICS:
# expected shape: [c, h, w]
# expected value range: [-1.0, 1.0]

def SSIM(pred, target):
    '''
    input type: np.ndarray (np.float)
    input shape: [c, h, w]
    input range: [-1.0, 1.0]
    '''
    pred, target = pred.transpose((1, 2, 0)), target.transpose((1, 2, 0))  # color channel is needed as last dim
    # this parametrisazion matches Wang et al.: "Image quality assessment: From error visibility to structural similarity."
    return ssim(target, pred, data_range=2, win_size=5, multichannel=True,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def PSNR(pred, target):
    '''
    input type: np.ndarray (np.float)
    input shape: [c, h, w]
    input range: [-1.0, 1.0]
    '''
    return psnr(target, pred, data_range=2)


def MSE(pred, target):
    '''
    input type: np.ndarray (np.float)
    input shape: [c, h, w]
    input range: [-1.0, 1.0]
    '''
    return np.mean((pred - target) ** 2, axis=0).sum()


def MAE(pred, target):
    '''
    input type: np.ndarray (np.float)
    input shape: [c, h, w]
    input range: [-1.0, 1.0]
    '''
    return np.mean(np.abs(pred - target), axis=0).sum()


lpips_alex = lpips.LPIPS(net='alex') # LPIPS ver. 0.1.4
def LPIPS(pred, target):
    '''
    input type: torch.tensor (torch.float)
    input shape: [c, h, w]
    input range: [-1.0, 1.0]
    '''
    return lpips_alex(target, pred)