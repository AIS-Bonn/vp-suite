import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset.dataset_utils import preprocess_img, preprocess_mask


class SynpickSegmentationDataset(Dataset):

    def __init__(self, data_dir, num_classes, augmentation=None):

        # searches for rgb and mask images in the given data dir
        images_dir = os.path.join(data_dir, 'rgb')
        masks_dir = os.path.join(data_dir, 'masks')
        self.image_ids = sorted(os.listdir(images_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))
        for a, b in zip(self.image_ids, self.mask_ids):
            if a[:-4] != b[:-4]:
                print(a, b)
                raise ValueError("image filenames are mask filenames do not match!")
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.mask_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        self.img_h, self.img_w, self.img_c = cv2.cvtColor(cv2.imread(self.image_fps[0]), cv2.COLOR_BGR2RGB).shape

        self.augmentation = augmentation
        self.num_classes = num_classes

    def __getitem__(self, i):

        image = cv2.cvtColor(cv2.imread(self.image_fps[i]), cv2.COLOR_BGR2RGB)
        mask = np.expand_dims(cv2.imread(self.mask_fps[i], 0), axis=-1)  # imread() mode 0 -> grayscale

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        image, mask = preprocess_img(image), preprocess_mask(mask)

        return image, mask

    def __len__(self):
        return len(self.image_ids)