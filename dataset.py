import os, time

import albumentations as albu
import numpy as np
import torch

import cv2

from torch.utils.data import Dataset

from utils import colorize_semseg

NUM_CLASSES = 22  # 21 YCB-Video objects and the background
CLASSES = ['object_{}'.format(i) for i in range(1, NUM_CLASSES)]

class SynpickSegmentationDataset(Dataset):

    def __init__(self, data_dir, augmentation=None):

        # searches for rgb and mask images in the given data dir
        images_dir = os.path.join(data_dir, 'rgb')
        masks_dir = os.path.join(data_dir, 'masks')
        self.image_ids = sorted(os.listdir(images_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))
        for a, b in zip(self.image_ids, self.mask_ids):
            if a[:-4] != b[:-4]:
                print(a, b)
                raise ValueError("image filenames are mask filenames do not match!")
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        self.augmentation = augmentation

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.expand_dims(cv2.imread(self.masks_fps[i], 0), axis=-1)  # imread() mode 0 -> grayscale

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        image, mask = preprocess_img(image), preprocess_mask(mask)

        return image, mask

    def __len__(self):
        return len(self.image_ids)


class SynpickVideoDataset(Dataset):

    def __init__(self, data_dir, vid_type:tuple, num_frames=8, step=1, allow_overlap=True):
        super(SynpickVideoDataset, self).__init__()

        self.is_mask = "mask" in vid_type[0]
        self.is_colorized_mask = self.is_mask and vid_type[1] == 3
        self.num_channels = NUM_CLASSES if self.is_mask and not self.is_colorized_mask else vid_type[1]
        self.data_ids = sorted(os.listdir(data_dir))
        self.data_fps = [os.path.join(data_dir, image_id) for image_id in self.data_ids]

        self.total_len = len(self.data_ids)
        self.step = step  # if >1, (step - 1) frames are skipped between each frame
        self.sequence_length = (num_frames - 1) * self.step + 1  # num_frames also includes prediction horizon
        # allow_overlap == True: Frames are packed into trajectories like [[0, 1, 2], [1, 2, 3], ...]. False: [[0, 1, 2], [3, 4, 5], ...]
        self.allow_overlap = allow_overlap

        # determine which dataset indices are valid for given sequence length T
        self.all_idx = []
        self.valid_idx = []
        for idx in range(len(self.data_ids)):
            # last T frames mustn't be chosen as the start of a sequence
            # -> declare indices of each trajectory's first T images as invalid and shift the indices back by T
            self.all_idx.append(idx)
            frame_num = int(self.data_ids[idx][-10:-4])
            frame_num_ok = frame_num >= self.sequence_length
            overlap_ok = self.allow_overlap or frame_num % self.sequence_length == 0
            if frame_num_ok and overlap_ok:
                self.valid_idx.append((idx - self.sequence_length) % self.total_len)

        # print(self.all_idx)
        # print(self.valid_idx)
        # exit(0)

        if len(self.valid_idx) < 1:
            raise ValueError("No valid indices in generated dataset! "
                             "Perhaps the calculated sequence length is longer than the trajectories of the data?")

    def __getitem__(self, i):
        true_i = self.valid_idx[i]
        frames = []
        for t in range(0, self.sequence_length, self.step):
            if self.is_colorized_mask:
                datapoint = preprocess_mask_colorize(cv2.imread(self.data_fps[true_i + t], 0), self.num_channels)
            elif self.is_mask:
                datapoint = np.expand_dims(cv2.imread(self.data_fps[true_i + t], 0), axis=-1)  # imread() grayscale mode
                datapoint = preprocess_mask_inflate(datapoint, self.num_channels) # apply preprocessing
            else:
                datapoint = cv2.imread(self.data_fps[true_i + t])
                datapoint = cv2.cvtColor(datapoint, cv2.COLOR_BGR2RGB)
                datapoint = preprocess_img(datapoint)  # apply preprocessing
            frames.append(datapoint)

        frames = torch.stack(frames, dim=0)  # [T, c, h, w]
        return frames

    def __len__(self):
        return len(self.valid_idx)

# ==============================================================================

def preprocess_mask(x):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))

def preprocess_mask_inflate(x, num_channels):
    x = torch.from_numpy(x.transpose(2, 0, 1))
    x_list = [(x == i) for i in range(num_channels)]
    return torch.cat(x_list, dim=0).float()

def preprocess_mask_colorize(x, num_channels):
    x = colorize_semseg(x.squeeze(), num_channels)
    return preprocess_img(x)

def postprocess_mask(x):
    return x.cpu().numpy().astype('uint8')

def preprocess_img(x):
    '''
    [0, 255, np.uint8] -> [-1, 1, torch.float32]
    '''
    torch_x = torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))
    return (2 * torch_x / 255) - 1

def postprocess_img(x):
    '''
    [~-1, ~1, torch.float32] -> [0, 255, np.uint8]
    '''
    scaled_x = (torch.clamp(x, -1, 1) + 1) * 255 / 2
    return scaled_x.cpu().numpy().astype('uint8')

# ==============================================================================

def synpick_seg_val_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(270, 480),
    ]
    return albu.Compose(test_transform)


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