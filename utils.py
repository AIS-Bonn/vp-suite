import albumentations as albu
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math, sys
from PIL import Image
import matplotlib.pyplot as plt
import hsluv

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