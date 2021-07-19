import os
import numpy as np
import torch

import cv2

from torch.utils.data import Dataset

class SynpickSegmentationDataset(Dataset):

    NUM_CLASSES = 22  # 21 YCB-Video objects and the background
    CLASSES = ['object_{}'.format(i) for i in range(1, NUM_CLASSES)]

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

    def __init__(self, data_dir, sequence_length=8):
        super(SynpickVideoDataset, self).__init__()

        self.sequence_length = sequence_length  # sequence length shall also include prediction horizon
        self.image_ids = sorted(os.listdir(data_dir))
        self.total_len = len(self.image_ids)

        self.images_fps = [os.path.join(data_dir, image_id) for image_id in self.image_ids]

        # determine which dataset indices are valid for given sequence length T
        self.all_idx = []
        self.valid_idx = []
        for idx in range(len(self.image_ids)):
            self.all_idx.append(idx)
            # last T frames mustn't be chosen as the start of a sequence
            # -> declare indices of each trajectory's first T images as invalid and shift the indices back by T
            frame_num = int(self.image_ids[idx][-10:-4])
            if frame_num >= self.sequence_length:
                self.valid_idx.append((idx - self.sequence_length) % self.total_len)

        if len(self.valid_idx) < 1:
            raise ValueError("No valid indices in generated dataset! "
                             "Perhaps the given sequence length is longer than the trajectories of the data?")

    def __getitem__(self, i):
        true_i = self.valid_idx[i]
        images = []
        for t in range(self.sequence_length):
            image = cv2.imread(self.images_fps[true_i + t])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # apply preprocessing
            image = preprocess_img(image)

            images.append(image)

        frames = torch.stack(images, dim=0)  # [T, c, h, w]
        return frames

    def __len__(self):
        return len(self.valid_idx)


def preprocess_mask(x):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))

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