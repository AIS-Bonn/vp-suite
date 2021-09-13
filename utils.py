from typing import List
import math

import hsluv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
import torch
import torch.linalg as linalg

def get_2_wasserstein_dist(pred, real):
    '''
    Calulates the two components of 2-Wasserstein metric:
    The general formula is given by: d(P_real, P_pred = min_{X, Y} E[|X-Y|^2]

    For multivariate gaussian distributed inputs x_real ~ MN(mu_real, cov_real) and x_pred ~ MN(mu_pred, cov_pred),
    this reduces to: d = |mu_real - mu_pred|^2 - Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))

    Fast method: https://arxiv.org/pdf/2009.14075.pdf

    Input shape: [b, n]
    Output shape: scalar
    '''

    if pred.shape != real.shape:
        raise ValueError("Expecting equal shapes for pred and real!")

    # the following ops need some extra precision
    pred, real = pred.transpose(0, 1).double(), real.transpose(0, 1).double()  # [n, b]
    mu_pred, mu_real = torch.mean(pred, dim=1, keepdim=True), torch.mean(real, dim=1, keepdim=True)  # [n, 1]
    n, b = pred.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_pred = pred - mu_pred
    E_real = real - mu_real
    cov_pred = torch.matmul(E_pred, E_pred.t()) * fact  # [n, n]
    cov_real = torch.matmul(E_real, E_real.t()) * fact

    # calculate Tr((cov_real * cov_pred)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues of the mm(cov_pred, cov_real) are real-valued, so for M, too.
    #  TODO further dive into mathematical intuition about why the eigenvalues are guaranteed to be real-valued
    C_pred = E_pred * math.sqrt(fact)  # [n, n], "root" of covariance
    C_real = E_real * math.sqrt(fact)
    M_l = torch.matmul(C_pred.t(), C_real)
    M_r = torch.matmul(C_real.t(), C_pred)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))
    trace_term = torch.trace(cov_pred + cov_real) - 2.0 * sq_tr_cov  # scalar

    # |mu_real - mu_pred|^2
    diff = mu_real - mu_pred  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()


def most(l: List[bool], factor=0.67):
    '''
    Like List.all(), but not 'all' of them.
    '''
    return sum(l) >= factor * len(l)


def get_grid_vis(input, mode='RGB'):

    val = input.detach().clone()
    b, c, h, w = val.shape
    grid_size = math.ceil(math.sqrt(b))
    imgmatrix = np.zeros((3 if mode == "RGB" else 1,
                          (h+1) * grid_size - 1,
                          (w+1) * grid_size - 1,))

    for i in range(b):
        x, y = i % grid_size, i // grid_size
        imgmatrix[:, y * (h+1) : (y+1)*(h+1)-1, x * (w+1) : (x+1)*(w+1)-1] = val[i]

    imgmatrix = imgmatrix.transpose((1, 2, 0)) if mode == "RGB" else imgmatrix.squeeze()
    return Image.fromarray((imgmatrix * 255).astype('uint8')).convert("RGB")


def save_seg_vis(out_fp, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(out_fp)


def get_color_array(color):
    if color == "green":
        color_array = np.array([0, 200, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    elif color == "red":
        color_array = np.array([150, 0, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    elif color == "yellow":
        color_array = np.array([100, 100, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        color_array = np.array([255, 255, 255], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    return color_array


def add_border_around_vid(vid, c_and_l, b_width=10):

    _, _, h, w = vid.shape
    color_bars_vertical = [np.tile(get_color_array(c), (l, 1, h, b_width)) for (c, l) in c_and_l]
    cbv = np.concatenate(color_bars_vertical, axis=0)

    color_bars_horizontal = [np.tile(get_color_array(c), (l, 1, b_width, w + 2 * b_width)) for (c, l) in c_and_l]
    cbh = np.concatenate(color_bars_horizontal, axis=0)

    vid = np.concatenate([cbv, vid, cbv], axis=-1)   # add bars in the width dim
    vid = np.concatenate([cbh, vid, cbh], axis=-2)   # add bars in the height dim
    return vid


def save_vid_vis(out_fp, video_in_length, **trajs):

    T, _, h, w = list(trajs.values())[0].shape
    T_in, T_pred = video_in_length, T-video_in_length
    for key, traj in trajs.items():
        if "true_" in key.lower() or "gt_" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("green", T)], b_width=16)
        elif "seg" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("yellow", T)], b_width=16)
        else:
            trajs[key] = add_border_around_vid(traj, [("green", T_in), ("red", T_pred)], b_width=16)

    n_trajs = len(trajs)
    plt_scale = 0.01
    plt_cols = math.ceil(math.sqrt(n_trajs))
    plt_rows = math.ceil(n_trajs / plt_cols)
    plt_w = 1.2 * w * plt_scale * plt_cols
    plt_h = 1.4 * h * plt_scale * plt_rows
    fig = plt.figure(figsize=(plt_w, plt_h), dpi=100)

    def update(t):
        for i, (name, traj) in enumerate(trajs.items()):
            plt.subplot(plt_rows, plt_cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(traj[t].transpose(1, 2, 0))

    anim = FuncAnimation(fig, update, frames=np.arange(T), interval=500)
    anim.save(out_fp, writer="imagemagick", dpi=200)
    plt.close(fig)


def colorize_semseg(input : np.ndarray, num_classes : int):
    '''
    Assigns a unique hue value to each class and replaces each pixel's class value with the corresponding RGB vector.
    The <num_classes> different hue values are spread out evenly over [0°, 360°).
    num_classes also counts the background.
    '''

    assert input.dtype == np.uint8
    input_shape = input.shape

    colors = np.zeros((num_classes, 3)).astype('uint8')
    colors[0] = [255, 255, 255]
    for i in range(1, num_classes):
        hue = (i-1) * 360.0 / (num_classes-1)
        rgb = hsluv.hsluv_to_rgb([hue, 100, 40])
        colors[i] = (np.array(rgb) * 255.0).astype('uint8')

    flattened = input.flatten()  # [-1]
    colorized = colors[flattened].reshape(*input_shape, 3)  # [*input_shape, 3]
    return colorized


def test():
    a, b, c = [np.random.randint(low=0, high=256, size=(12, 3, 270, 480)).astype('uint8')] * 3
    save_vid_vis("out/test_clip.gif", 8, true_trajectory=a, pred1=b, pred2=c)


if __name__ == '__main__':
    test()