import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from vp_suite.utils.utils import most
from vp_suite.dataset._base_dataset import BaseVPDataset, VPData

class SynpickVideoDataset(BaseVPDataset):

    MAX_SEQ_LEN = 90  # a trajectory in the SynPick dataset is at least 90 frames
    NAME = "SynPick bin picking"
    ACTION_SIZE = 3
    DEFAULT_FRAME_SHAPE = (135, 240, 3)
    SKIP_FIRST_N = 72
    VALID_SPLITS = ["train", "val", "test"]

    def __init__(self, cfg, split):
        super(SynpickVideoDataset, self).__init__(cfg, split)

        self.data_dir = str(Path(cfg.data_dir) / split)
        images_dir = os.path.join(self.data_dir, 'rgb')
        scene_gt_dir = os.path.join(self.data_dir, 'scene_gt')

        self.image_ids = sorted(os.listdir(images_dir))
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]

        scene_gt_fps = [os.path.join(scene_gt_dir, scene_gt_fp) for scene_gt_fp in sorted(os.listdir(scene_gt_dir))]
        self.gripper_pos = {}
        for scene_gt_fp, ep in zip(scene_gt_fps, [int(a[-20:-14]) for a in scene_gt_fps]):
            with open(scene_gt_fp, "r") as scene_json_file:
                ep_dict = json.load(scene_json_file)
            gripper_pos = [ep_dict[frame_num][-1]["cam_t_m2c"] for frame_num in ep_dict.keys()]
            self.gripper_pos[ep] = gripper_pos
        self.total_len = len(self.image_ids)

        # determine which dataset indices are valid for given sequence length T
        self.all_idx = []
        self.valid_idx = []
        last_valid_idx = -1 * self.seq_len
        for idx in range(self.total_len - self.seq_len + 1):

            self.all_idx.append(idx)
            ep_nums = [self.ep_num_from_id(self.image_ids[idx + offset]) for offset in self.frame_offsets]
            frame_nums = [self.frame_num_from_id(self.image_ids[idx + offset]) for offset in self.frame_offsets]

            # first few frames are discarded
            if frame_nums[0] < self.SKIP_FIRST_N:
                continue

            # last T frames of an episode mustn't be chosen as the start of a sequence
            if ep_nums[0] != ep_nums[-1]:
                continue

            # overlap is not allowed -> sequences should not overlap
            if idx < last_valid_idx + self.seq_len:
                continue

            # discard sequences without considerable gripper movement
            gripper_pos = [self.gripper_pos[ep_nums[0]][frame_num] for frame_num in frame_nums]
            gripper_pos_deltas = self.get_gripper_pos_xydist(gripper_pos)
            gripper_pos_deltas_above_min = [(delta > 1.0) for delta in gripper_pos_deltas]
            gripper_pos_deltas_below_max = [(delta < 30.0) for delta in gripper_pos_deltas]
            gripper_movement_ok = most(gripper_pos_deltas_above_min) and all(gripper_pos_deltas_below_max)
            if not gripper_movement_ok:
                continue

            self.valid_idx.append(idx)
            last_valid_idx = idx

        if len(self.valid_idx) < 1:
            raise ValueError("No valid indices in generated dataset! "
                             "Perhaps the calculated sequence length is longer than the trajectories of the data?")

    def __getitem__(self, i) -> VPData:

        i = self.valid_idx[i]  # only consider valid indices
        idx = range(i, i + self.seq_len, self.seq_step)  # create range of indices for frame sequence

        ep_num = self.ep_num_from_id(self.image_ids[idx[0]])
        frame_nums = [self.frame_num_from_id(self.image_ids[id_]) for id_ in idx]
        gripper_pos = [self.gripper_pos[ep_num][frame_num] for frame_num in frame_nums]
        actions = torch.from_numpy(self.get_gripper_pos_diff(gripper_pos)).float()  # [t, a] sequence length is one less!

        imgs_ = [cv2.cvtColor(cv2.imread(self.image_fps[id_]), cv2.COLOR_BGR2RGB) for id_ in idx]
        imgs = [self.preprocess_img(img) for img in imgs_]
        rgb = torch.stack(imgs, dim=0)  # [t, c, h, w]

        data = { "frames": rgb, "actions": actions }
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

# === SynPick data preparation tools ===========================================


TRAIN_VAL_SPLIT = (7/8)  # train:val  7:1

def prepare_synpick(cfg):

    path = Path(cfg.in_path)
    seed = cfg.seed

    random.seed(seed)

    train_path = path / "train"
    test_path = path / "test"

    # get all training image FPs for rgb
    rgbs = sorted(train_path.glob("*/rgb/*.jpg"))
    segs = sorted(train_path.glob("*/class_index_masks/*.png"))
    scene_gts = sorted(train_path.glob("*/scene_gt.json"))

    num_ep = int(Path(rgbs[-1]).parent.parent.stem) + 1
    train_eps = [i for i in range(num_ep)]
    random.shuffle(train_eps)
    cut = int(num_ep * TRAIN_VAL_SPLIT)
    train_eps = train_eps[:cut]

    # split rgb files into train and val by episode number only , as we need contiguous motions for video
    train_rgbs, val_rgbs, train_segs, val_segs, train_scene_gts, val_scene_gts = [], [], [], [], [], []
    for rgb, seg in zip(rgbs, segs):
        ep = int(Path(rgb).parent.parent.stem) + 1
        if ep in train_eps:
            train_rgbs.append(rgb)
            train_segs.append(seg)
        else:
            val_rgbs.append(rgb)
            val_segs.append(seg)

    for scene_gt in scene_gts:
        ep = int(Path(scene_gt).parent.stem) + 1
        if ep in train_eps:
            train_scene_gts.append(scene_gt)
        else:
            val_scene_gts.append(scene_gt)

    test_rgbs = sorted(test_path.glob("*/rgb/*.jpg"))
    test_segs = sorted(test_path.glob("*/class_index_masks/*.png"))
    test_scene_gts = sorted(test_path.glob("*/scene_gt.json"))

    all_img_fps = [train_rgbs, train_segs, val_rgbs, val_segs, test_rgbs, test_segs]
    all_scene_gts = [train_scene_gts, val_scene_gts, test_scene_gts]
    out_path = Path("data").absolute() / f"vid_{path.stem}_{cfg.timestamp}"
    out_path.mkdir(parents=True)

    copy_synpick_imgs(all_img_fps, out_path, cfg.resize_ratio)
    copy_synpick_scene_gts(all_scene_gts, out_path)


def copy_synpick_imgs(all_fps, out_path, resize_ratio):

    # prepare and execute file copying
    all_out_paths = [(out_path / "train" / "rgb"), (out_path / "train" / "masks"),
                     (out_path / "val" / "rgb"), (out_path / "val" / "masks"),
                     (out_path / "test" / "rgb"), (out_path / "test" / "masks")]

    for op in all_out_paths:
        op.mkdir(parents=True)

    # copy files to new folder structure
    for fps, op in zip(all_fps, all_out_paths):
        for fp in tqdm(fps, postfix=op.parent.stem + "/" + op.stem):
            fp = Path(fp)
            img = cv2.imread(str(fp.absolute()), cv2.IMREAD_COLOR if "jpg" in fp.suffix else cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
            ep_number = ''.join(filter(str.isdigit, fp.parent.parent.stem)).zfill(6)
            out_fp = "{}_{}{}".format(ep_number, fp.stem, ".".join(fp.suffixes))
            cv2.imwrite(str((op / out_fp).absolute()), resized_img)


def copy_synpick_scene_gts(all_fps, out_path):

    # prepare and execute file copying
    all_out_paths = [(out_path / "train" / "scene_gt"), (out_path / "val" / "scene_gt"),
                     (out_path / "test" / "scene_gt")]

    for op in all_out_paths:
        op.mkdir(parents=True)

    # copy files to new folder structure
    for fps, op in zip(all_fps, all_out_paths):
        for fp in fps:
            ep_number = ''.join(filter(str.isdigit, fp.parent.stem)).zfill(6)
            out_fp = op / "{}_{}{}".format(ep_number, fp.stem, ".".join(fp.suffixes))
            print(fp, out_fp)
            shutil.copyfile(fp, out_fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="prepare_synpick")
    parser.add_argument("--in-path", type=str, help="directory for synpick data")
    parser.add_argument("--seed", type=int, default=42, help="rng seed for train/val split")
    parser.add_argument("--resize-ratio", type=float, default=0.125, help="Scale frame sizes by this amount")
    cfg = parser.parse_args()
    cfg.timestamp = int(time.time())
    prepare_synpick(cfg)
