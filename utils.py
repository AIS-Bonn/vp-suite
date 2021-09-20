from typing import List
import math

import hsluv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image

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


def save_vid_vis(out_fp, vid_input_length, **trajs):

    T, _, h, w = list(trajs.values())[0].shape
    T_in, T_pred = vid_input_length, T-vid_input_length
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

    # 1 value less since background is painted white
    hues = [360.0*i / (num_classes-1) for i in range(num_classes-1)]
    # arrange hue values in star pattern so that neighboring classes values lead to different hues
    # e.g. [1, 2, 3, 4, 5, 6, 7] -> [1, 5, 2, 6, 3, 7, 4]
    if len(hues) % 2 == 0:  # even length
        hues = [item for pair in list(zip(hues[:len(hues)//2], hues[len(hues)//2:])) for item in pair]
    else:  # odd length -> append element to make it even and remove that element after rearrangement
        hues.append[None]
        hues = [item for pair in list(zip(hues[:len(hues)//2], hues[len(hues)//2:])) for item in pair]
        hues.pop()

    colors = np.zeros((num_classes, 3)).astype('uint8')
    colors[0] = [255, 255, 255]
    for i in range(1, num_classes):
        rgb = hsluv.hsluv_to_rgb([hues[i-1], 100, 40])
        colors[i] = (np.array(rgb) * 255.0).astype('uint8')

    flattened = input.flatten()  # [-1]
    colorized = colors[flattened].reshape(*input_shape, 3)  # [*input_shape, 3]
    return colorized


def test():
    a, b, c = [np.random.randint(low=0, high=256, size=(12, 3, 270, 480)).astype('uint8')] * 3
    save_vid_vis("out/test_clip.gif", 8, true_trajectory=a, pred1=b, pred2=c)


if __name__ == '__main__':
    test()