import sys, os, time, json, math

import cv2
import albumentations as albu
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


def preprocess_mask(x):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))

def preprocess_mask_inflate(x, num_channels):
    # input: [h, w] or [t, h, w], np.uint8
    x = torch.from_numpy(x.squeeze())  # [1, h, w]
    x_list = [(x == i) for i in range(num_channels)]
    return torch.stack(x_list, dim=0).float()

def preprocess_mask_colorize(x, num_channels):
    x = colorize_semseg(x, num_channels)
    return preprocess_img(x)

def postprocess_mask(x):
    return x.cpu().numpy().astype('uint8')

def preprocess_img(x):
    '''
    [0, 255, np.uint8] -> [-1, 1, torch.float32]
    '''
    permutation = (2, 0, 1) if x.ndim == 3 else (0, 3, 1, 2)
    torch_x = torch.from_numpy(x.transpose(permutation).astype('float32'))
    return (2 * torch_x / 255) - 1

def postprocess_img(x):
    '''
    [~-1, ~1, torch.float32] -> [0, 255, np.uint8]
    '''
    scaled_x = (torch.clamp(x, -1, 1) + 1) * 255 / 2
    return scaled_x.cpu().numpy().astype('uint8')

# ==============================================================================

def synpick_seg_val_augmentation(img_h=270):
    test_transform = [
        albu.PadIfNeeded(img_h, img_h * 16 // 9),
    ]
    return albu.Compose(test_transform)


def synpick_seg_train_augmentation(img_h=270):

    img_w_p2floored = 2 ** int(np.floor(np.log2(img_h)))
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=img_h, min_width=img_h, always_apply=True, border_mode=0),
        albu.RandomCrop(height=img_w_p2floored, width=img_w_p2floored, always_apply=True),
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