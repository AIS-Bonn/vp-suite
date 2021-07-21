import albumentations as albu
import numpy as np
from tqdm import tqdm
import torch
import torch.linalg as linalg
import torch.nn.functional as F
import torch.nn as nn
import math, sys
from PIL import Image
import matplotlib.pyplot as plt
import hsluv
import cv2
from moviepy.editor import ImageSequenceClip

from config import DEVICE


def symmat_sqrt(matrix, eps=1e-10):
    '''
    Computes the square root A of a symmetric matrix X such that AA = X.
    Only works for symmetric matrices!

    Source: https://github.com/pytorch/pytorch/issues/25481#issuecomment-576493693
    '''

    try:
        _, s, vh = linalg.svd(matrix)
        cond = np.linalg.cond(matrix.detach().cpu().numpy())
        print(f"cond: {cond}")
    except RuntimeError as e:
        print(matrix.shape)
        print(f"SVD fails, cond: {np.linalg.cond(matrix.detach().cpu().numpy())}")
        exit(0)
    v = vh.transpose(-2, -1).conj()

    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def get_2_wasserstein_dist(pred, real, fast_method=False):
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

    # calculate Tr((cov_real * cov_pred)^(1/2)).
    if fast_method:

        # calc. with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
        C_pred = E_pred * math.sqrt(fact)  # [n, n], "root" of covariance
        C_real = E_real * math.sqrt(fact)

        M_l = torch.matmul(C_pred.t(), C_real)
        M_r = torch.matmul(C_real.t(), C_pred)
        M = torch.matmul(M_l, M_r)

        # TODO can we assume that M is symmetric (meaning all eigenvals are real)? Or can we take abs() of complex eigenvals?
        S = linalg.eigvals(M)
        sq_tr_cov = S.sqrt().abs().sum()
    else:

        # calc. by first by using Tr((XY)^(1/2)) = Tr((AYA)^(1/2)) with AA =
        # As cov_real and A * cov_pred * A are symmetric, their root matrix can be found using symmat_sqrt().
        # TODO symmat_sqrt() uses svd(), which does not always work! (np.svd fails at the same step)
        #  (Is it bc. given matrix might be ill-conditioned/near-singular?)
        #  (Might also be due to some low-level lib errors that are used by np/torch.svd(). In that case simply use the fast method)
        A = symmat_sqrt(cov_real)
        A_cov_pred_A = torch.matmul(A, torch.matmul(cov_pred, A))
        sq_tr_cov = torch.trace(symmat_sqrt(A_cov_pred_A))  # scalar

    # plug the sqrt_trace_component into Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))
    trace_term = torch.trace(cov_pred + cov_real) - 2.0 * sq_tr_cov  # scalar

    # |mu_real - mu_pred|^2
    diff = mu_real - mu_pred  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together#
    return (trace_term + mean_term).float()


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


# helper function for data visualization
def save_vis(out_fp, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(out_fp)


def save_video_vis(out_fp, video_in_length, **trajs):

    # put green bars next to GT trajectory
    gt_traj = trajs["true_trajectory"]
    T, _, h, w = gt_traj.shape
    green = np.array([0, 200, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    gt_traj_bar = np.tile(green, (T, 1, h, 20))  # [T, 3, h, 20]
    out_barred = np.concatenate([gt_traj_bar, gt_traj, gt_traj_bar], axis=-1)  # add bars in the width dim

    # put green bars that turn red for pred. frames next to predicted trajectories and concat the 4D arrays depth-wise
    if len(trajs) > 1:
        red = np.array([150, 0, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
        black = np.array([0, 0, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
        black_bar = np.tile(black, (T, 1, h, 10))  # [T, 3, h, 10]
        green_and_red = [np.tile(green, (video_in_length, 1, h, 20)), np.tile(red, (T-video_in_length, 1, h, 20))]
        pr_traj_bar = np.concatenate(green_and_red, axis=0)   # [T, 3, h, 20]

        for _, (name, pr_traj) in enumerate(trajs.items()):
            if name == "true_trajectory": continue
            pr_traj_barred = np.concatenate([black_bar, pr_traj_bar, pr_traj, pr_traj_bar], axis=-1)  # add bars in the width dim
            out_barred = np.concatenate([out_barred, pr_traj_barred], axis=-1)  # add bars in the width dim

    out_frames, _, out_h, out_w = out_barred.shape
    out_barred = np.transpose(out_barred, (0, 2, 3, 1))  # [T, h, w, 3]

    out_FPS = 2
    clip = ImageSequenceClip(list(out_barred), fps=out_FPS)
    clip.write_gif(out_fp, fps=out_FPS, logger=None)


def colorize_semseg(input : np.ndarray, num_classes : int):
    # num_classes also counts the background
    assert input.ndim == 2  # input: [h, w]
    assert input.dtype == np.uint8
    h, w = input.shape

    colors = np.zeros((num_classes, 3)).astype('uint8')
    colors[0] = [255, 255, 255]
    for i in range(1, num_classes):
        hue = (i-1) * 360.0 / (num_classes-1)
        rgb = hsluv.hsluv_to_rgb([hue, 100, 40])
        colors[i] = (np.array(rgb) * 255.0).astype('uint8')

    flattened = input.astype('uint8').reshape(-1)  # [h*w]
    colorized = colors[flattened]  # [h*w, 3]
    return colorized.reshape(h, w, 3)


class CrossEntropyLoss():
    '''
    b means batch_size.
    n means num_classes, including background.
    '''
    def __init__(self):
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def get_loss(self, output: torch.Tensor, target: torch.tensor):

        # assumed shapes [b, n, h, w] for output, [b, 1, h, w] for target
        batch_size, num_classes, h, w = output.shape
        output = output.permute((0, 2, 3, 1)).reshape(-1, num_classes)  # shape: [b*h*w, n+1]
        target = target.view(-1).long()  # [b*h*w]

        return self.loss(output, target)  # scalar, reduced along b*h*w


def get_accuracy(loader, seg_model, device):
    num_correct = 0
    num_pixels = 0
    seg_model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # shapes: [1, 3, h, w] for x and [1, h, w] for y
            preds = torch.argmax(seg_model(x), dim=1)   #  [1, h, w]
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    seg_model.train()

    return 100.0 * num_correct / num_pixels


def validate_video_model(loader, pred_model, device, video_in_length, video_pred_length, loss_fn, concat_input_for_loss):

    pred_model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        val_total_loss = []
        for batch_idx, data in enumerate(loop):
            data = data.to(device)  # [b, T, h, w], with T = in_length + pred_length
            input = data[:, :video_in_length]
            targets = data if concat_input_for_loss else data[:, video_in_length:]
            predictions = pred_model.pred_n(input, pred_length=video_pred_length)
            if concat_input_for_loss:
                predictions = torch.cat([input, predictions], dim=1)
            val_total_loss.append(loss_fn(predictions, targets).item())
    pred_model.train()

    mean_loss = sum(val_total_loss) / len(val_total_loss)
    return mean_loss

def synpick_seg_train_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=270, min_width=270, always_apply=True, border_mode=0),
        albu.RandomCrop(height=256, width=256, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def synpick_seg_val_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(270, 480),
    ]
    return albu.Compose(test_transform)


def test():
    a, b, c = [np.random.randint(low=0, high=256, size=(12, 3, 270, 480)).astype('uint8')] * 3
    save_video_vis("out/test_clip.gif", 8, true_trajectory=a, pred1=b, pred2=c)

if __name__ == '__main__':
    test()