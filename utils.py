import numpy as np
from tqdm import tqdm
import torch
import torch.linalg as linalg
import math
from PIL import Image
import matplotlib.pyplot as plt
import hsluv
from moviepy.editor import ImageSequenceClip


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
    S = linalg.eigvals(M)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))
    trace_term = torch.trace(cov_pred + cov_real) - 2.0 * sq_tr_cov  # scalar

    # |mu_real - mu_pred|^2
    diff = mu_real - mu_pred  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
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


def save_vid_vis(out_fp, video_in_length, **trajs):

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


def validate_seg_model(loader, seg_model, device):
    num_correct = 0
    num_pixels = 0
    seg_model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # shapes: [1, 3, h, w] for x and [1, h, w] for y
            preds = torch.argmax(seg_model(x), dim=1)   # [1, h, w]
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    seg_model.train()

    return 100.0 * num_correct / num_pixels


def validate_vid_model(loader, pred_model, device, video_in_length, video_pred_length, losses):

    pred_model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        all_losses = {key: [] for key in losses.keys()}
        for batch_idx, data in enumerate(loop):

            # fwd
            data = data.to(device)  # [b, T, h, w], with T = video_tot_length
            input, targets = data[:, :video_in_length], data[:, video_in_length:]
            predictions = pred_model.pred_n(input, pred_length=video_pred_length)

            # metrics
            predictions_full = torch.cat([input, predictions], dim=1)
            targets_full = data
            for name, (loss_fn, use_full_input, _) in losses.items():
                pred = predictions_full if use_full_input else predictions
                real = targets_full if use_full_input else targets
                loss = loss_fn(pred, real).item()
                all_losses[name].append(loss)

    pred_model.train()

    print("Validation losses:")
    for key in all_losses.keys():
        cur_losses = all_losses[key]
        avg_loss = sum(cur_losses) / len(cur_losses)
        print(f" - {key}: {avg_loss}")
        all_losses[key] = avg_loss

    return all_losses


def test():
    a, b, c = [np.random.randint(low=0, high=256, size=(12, 3, 270, 480)).astype('uint8')] * 3
    save_vid_vis("out/test_clip.gif", 8, true_trajectory=a, pred1=b, pred2=c)

if __name__ == '__main__':
    test()