import os, time, json, math

import albumentations as albu
import numpy as np
import torch
import torch.nn.functional as F

import cv2

from torch.utils.data import Dataset

from utils import colorize_semseg, most
from config import SYNPICK_CLASSES

class SynpickSegmentationDataset(Dataset):

    def __init__(self, data_dir, augmentation=None, num_classes=SYNPICK_CLASSES):

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


class SynpickVideoDataset(Dataset):

    def __init__(self, data_dir, num_frames=8, step=1, allow_overlap=True, num_classes=SYNPICK_CLASSES):
        super(SynpickVideoDataset, self).__init__()

        images_dir = os.path.join(data_dir, 'rgb')
        masks_dir = os.path.join(data_dir, 'masks')
        scene_gt_dir = os.path.join(data_dir, 'scene_gt')

        self.include_gripper = num_classes > SYNPICK_CLASSES
        self.check_gripper_movement = self.include_gripper and os.path.isdir(scene_gt_dir)

        self.image_ids = sorted(os.listdir(images_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))
        for a, b in zip(self.image_ids, self.mask_ids):
            if a[:-4] != b[:-4]:
                print(a, b)
                raise ValueError("image filenames are mask filenames do not match!")

        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.mask_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        if self.check_gripper_movement:
            scene_gt_fps = [os.path.join(scene_gt_dir, scene_gt_fp) for scene_gt_fp in sorted(os.listdir(scene_gt_dir))]
            self.gripper_pos = {}
            for scene_gt_fp, ep in zip(scene_gt_fps, [int(a[-20:-14]) for a in scene_gt_fps]):
                with open(scene_gt_fp, "r") as scene_json_file:
                    ep_dict = json.load(scene_json_file)
                gripper_pos = [ep_dict[frame_num][-1]["cam_t_m2c"] for frame_num in ep_dict.keys()]
                self.gripper_pos[ep] = gripper_pos

        self.skip_first_n = 72
        self.total_len = len(self.image_ids)
        self.step = step  # if >1, (step - 1) frames are skipped between each frame
        self.sequence_length = (num_frames - 1) * self.step + 1  # num_frames also includes prediction horizon
        self.frame_offsets = range(0, num_frames * self.step, self.step)

        # If allow_overlap == True: Frames are packed into trajectories like [[0, 1, 2], [1, 2, 3], ...]. False: [[0, 1, 2], [3, 4, 5], ...]
        self.allow_overlap = allow_overlap
        self.num_classes = num_classes
        self.action_size = 3

        # determine which dataset indices are valid for given sequence length T
        self.all_idx = []
        self.valid_idx = []
        last_valid_idx = -1 * self.sequence_length
        for idx in range(len(self.image_ids) - self.sequence_length + 1):

            self.all_idx.append(idx)
            ep_nums = [self.ep_num_from_id(self.image_ids[idx + offset]) for offset in self.frame_offsets]
            frame_nums = [self.frame_num_from_id(self.image_ids[idx + offset]) for offset in self.frame_offsets]

            # first few frames are discarded
            if frame_nums[0] < self.skip_first_n:
                continue

            # last T frames of an episode mustn't be chosen as the start of a sequence
            if ep_nums[0] != ep_nums[-1]:
                continue

            # if overlap is not allowed, sequences should not overlap
            if not self.allow_overlap and idx < last_valid_idx + self.sequence_length:
                continue

            # if gripper positions are included, discard sequences without considerable gripper movement
            if self.check_gripper_movement:
                gripper_pos = [self.gripper_pos[ep_nums[0]][frame_num] for frame_num in frame_nums]
                gripper_pos_deltas = self.get_gripper_pos_xydist(gripper_pos)
                gripper_pos_deltas_above_min = [(delta > 1.0) for delta in gripper_pos_deltas]
                gripper_pos_deltas_below_max = [(delta < 30.0) for delta in gripper_pos_deltas]
                gripper_movement_ok = most(gripper_pos_deltas_above_min) and all(gripper_pos_deltas_below_max)
                if not gripper_movement_ok:
                    continue

            self.valid_idx.append(idx)
            last_valid_idx = idx

        self.img_shape = cv2.cvtColor(cv2.imread(self.image_fps[self.valid_idx[0]]), cv2.COLOR_BGR2RGB).shape[:-1]

        #print(len(self.all_idx))
        #print(len(self.valid_idx))
        #exit(0)

        if len(self.valid_idx) < 1:
            raise ValueError("No valid indices in generated dataset! "
                             "Perhaps the calculated sequence length is longer than the trajectories of the data?")

    def __getitem__(self, i):

        i = self.valid_idx[i]  # only consider valid indices
        idx = range(i, i + self.sequence_length, self.step)  # create range of indices for frame sequence

        ep_num = self.ep_num_from_id(self.image_ids[idx[0]])
        frame_nums = [self.frame_num_from_id(self.image_ids[id_]) for id_ in idx]
        gripper_pos = [self.gripper_pos[ep_num][frame_num] for frame_num in frame_nums]
        actions = torch.from_numpy(self.get_gripper_pos_diff(gripper_pos)).float()
        # front-pad time dim with 0s -> 1st real action in 2nd entry (input together with 2nd frame)
        actions = F.pad(actions, (0, 0, 1, 0))

        imgs_ = [cv2.cvtColor(cv2.imread(self.image_fps[id_]), cv2.COLOR_BGR2RGB) for id_ in idx]
        masks_ = [cv2.imread(self.mask_fps[id_], 0) for id_ in idx]

        imgs = [preprocess_img(img) for img in imgs_]
        masks = [preprocess_mask_inflate(np.expand_dims(mask, axis=2), self.num_classes) for mask in masks_]
        colorized_masks = [preprocess_mask_colorize(mask, self.num_classes) for mask in masks_]

        data = {
            "rgb": torch.stack(imgs, dim=0),
            "mask": torch.stack(masks, dim=0),
            "colorized": torch.stack(colorized_masks, dim=0),
            "actions": actions
        }

        return data

    def __len__(self):
        return len(self.valid_idx)

    def comp_gripper_pos(self, old, new):
        x_diff, y_diff = new[0] - old[0], new[1] - old[1]
        return math.sqrt(x_diff * x_diff + y_diff * y_diff)

    def get_gripper_pos_xydist(self, gripper_pos):
        return [self.comp_gripper_pos(old, new) for old, new in zip(gripper_pos, gripper_pos[1:])]

    def get_gripper_pos_diff(self, gripper_pos):
        gripper_pos_numpy = np.array(gripper_pos)
        return np.stack([new-old for old, new in zip(gripper_pos_numpy, gripper_pos_numpy[1:])], axis=0)

    def ep_num_from_id(self, file_id: str):
        return int(file_id[-17:-11])

    def frame_num_from_id(self, file_id: str):
        return int(file_id[-10:-4])

# ==============================================================================

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