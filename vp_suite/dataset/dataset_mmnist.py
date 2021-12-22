import sys, os
sys.path.append("")

from pathlib import Path

import numpy as np

import torch
from vp_suite.dataset.base_dataset import BaseVPDataset, VPData

class MovingMNISTDataset(BaseVPDataset):

    MAX_SEQ_LEN = 20  # default MMNIST sequences span 20 frames
    NAME = "Moving MNIST"
    ACTION_SIZE = 0
    DEFAULT_FRAME_SHAPE = (64, 64, 3)
    TRAIN_KEEP_RATIO = 0.96  # big dataset -> val can be smaller

    def __init__(self, cfg, split):
        super(MovingMNISTDataset, self).__init__(cfg, split)

        self.data_dir = str(Path(cfg.data_dir) / split)
        self.data_ids = sorted(os.listdir(self.data_dir))
        self.data_fps = [os.path.join(self.data_dir, image_id) for image_id in self.data_ids]

    def __len__(self):
        return len(self.data_fps)

    def __getitem__(self, i) -> VPData:

        rgb_raw = np.load(self.data_fps[i])  # [t', h, w]
        rgb_raw = np.expand_dims(rgb_raw, axis=-1).repeat(3, axis=-1) # [t', h, w, c]
        rgb = self.preprocess_img(rgb_raw)
        rgb = rgb[:self.seq_len:self.seq_step]  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = { "frames": rgb, "actions": actions }
        return data

# === MMNIST data preparation tools ============================================


def split_big_mmnist_file(file_path, out_dir):
    data = np.load(file_path)["arr_0"].squeeze()
    num_traj = 60000 if "train" in file_path else 10000
    num_frames = data.shape[0] // num_traj
    data = data.reshape(num_traj, num_frames, 64, 64)
    print(data.shape)
    for i in range(data.shape[0]):
        cur_out_fp = Path(out_dir) / f"seq_{i:05d}.npy"
        np.save(cur_out_fp, data[i])


if __name__ == '__main__':
    file_path, out_dir = sys.argv[1], sys.argv[2]
    split_big_mmnist_file(file_path, out_dir)

    # data_dir = sys.argv[1]
    # dataset = MovingMNISTDataset(data_dir)
    # dp = dataset[0]
    # print(dp.shape, dp.min(), dp.max())