import os
import numpy as np
import torch

import cv2
import albumentations as albu

from torch.utils.data import Dataset

class SynpickDataset(Dataset):

    NUM_CLASSES = 22  # 21 YCB-Video objects and the background
    CLASSES = ['object_{}'.format(i) for i in range(1, NUM_CLASSES)]

    def __init__(self, data_dir, augmentation=None):
        images_dir = os.path.join(data_dir, 'rgb')
        masks_dir = os.path.join(data_dir, 'masks')
        self.image_ids = sorted(os.listdir(images_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))
        for a, b in zip(self.image_ids, self.mask_ids):
            if a != b: raise ValueError("A!=B")
        print("COMPLETE")
        exit(0)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        self.augmentation = augmentation
        self.is_val_dataset = is_val_dataset

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.expand_dims(cv2.imread(self.masks_fps[i], 0), axis=-1)  # imread() mode 0 -> grayscale

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        image, mask = to_torch(image), to_torch(mask)

        return image, mask

    def __len__(self):
        return len(self.image_ids)


def get_training_augmentation():
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


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(270, 480),
    ]
    return albu.Compose(test_transform)


def to_torch(x, **kwargs):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))