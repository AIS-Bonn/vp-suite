import sys, os
sys.path.append(".")

from pathlib import Path

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from tfrecord.tools.tfrecord2idx import create_index
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm


class BAIRPushingDataset(Dataset):

    def __init__(self, data_dir):

        self.obs_ids = [fn for fn in sorted(os.listdir(data_dir)) if str(fn).endswith("obs.npy")]
        self.actions_ids = [fn for fn in sorted(os.listdir(data_dir)) if str(fn).endswith("actions.npy")]

        if len(self.obs_ids) != len(self.actions_ids):
            raise ValueError("Different number of obs and action files found -> Delete dataset and prepare again!")
        elif len(self.obs_ids) == 0:
            raise ValueError("No trajectory files (.npy) found! Maybe you forgot to prepare the dataset?")

        self.obs_fps = [os.path.join(data_dir, i) for i in self.obs_ids]
        self.actions_fps = [os.path.join(data_dir, i) for i in self.actions_ids]

        self.seq_length = 30  # a trajectory in the BAIR robot pushing dataset is 30 timesteps

        self.channels = 3
        self.action_size = 4
        self.img_shape = np.load(self.obs_fps[0]).shape[1:3]  # h, w

    def __len__(self):
        return len(self.obs_fps)

    def __getitem__(self, i):

        rgb = np.load(self.obs_fps[i]).transpose((0, 3, 1, 2))  # [t, c, h, w]
        # convert entries range from [0, 255, np.uint8] to [-1, 1, torch.float32]
        rgb = torch.from_numpy(rgb).float()
        rgb = (2 * rgb / 255) - 1
        actions = torch.from_numpy(np.load(self.actions_fps[i])).float()

        data = {
            "rgb": rgb,
            "actions": actions
        }
        return data


def split_bair_traj_files(data_dir, delete_tfrecords):
    bair_ep_length = 30

    data_dir = Path(data_dir)
    data_files = [fn for fn in sorted(os.listdir(data_dir)) if str(fn).endswith(".tfrecords")]
    ep_number = 0
    for df in tqdm(data_files):
        tfr_fp = str((data_dir / df).resolve())
        index_fp = tfr_fp + ".index"
        if not os.path.isfile(index_fp):
            create_index(tfr_fp, index_fp)
        data_list = list(TFRecordDataset(tfr_fp, index_fp))
        for ep_i in range(len(data_list)):
            observations, actions = [], []
            for step_i in range(bair_ep_length):
                obs = np.array(data_list[ep_i][str(step_i) + '/image_aux1/encoded']).reshape(1, 64, 64, 3)
                # obs = np.array(data_list[ep][str(step) + '/image_main/encoded']).reshape(1, 64, 64, 3)
                action = np.array(data_list[ep_i][str(step_i) + '/action'])[np.newaxis, ...]

                observations.append(obs)
                actions.append(action)

            observations = np.concatenate(observations, axis=0)
            out_obs_fp = Path(data_dir) / f"seq_{ep_number:05d}_obs.npy"
            np.save(out_obs_fp, observations)

            actions = np.concatenate(actions, axis=0)
            out_actions_fp = Path(data_dir) / f"seq_{ep_number:05d}_actions.npy"
            np.save(out_actions_fp, actions)

            ep_number += 1
        if delete_tfrecords:
            os.remove(tfr_fp)
            os.remove(index_fp)


def prepare_bair(data_dir, delete_tfrecords):
    train_dir = os.path.join(data_dir, "softmotion30_44k", "train")
    test_dir = os.path.join(data_dir, "softmotion30_44k", "test")
    split_bair_traj_files(train_dir, delete_tfrecords)
    split_bair_traj_files(test_dir, delete_tfrecords)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    prepare_bair(data_dir, delete_tfrecords=True)
    main_dataset = BAIRPushingDataset(os.path.join(data_dir, "softmotion30_44k", "train"))
    test_dataset = BAIRPushingDataset(os.path.join(data_dir, "softmotion30_44k", "test"))
    obs, actions = main_dataset[0]
    print(len(main_dataset), len(test_dataset), obs.shape, actions.shape)


'''
def load_BAIR_pushing_dataset(dataset_fname, experience_size):

    for i in tqdm(range(len(tfr_files))):
        cur_tfr_file = tfr_files[i]
        tfr_fp = str((dataset_fp / cur_tfr_file).resolve())
        index_fp = tfr_fp + ".index"
        if not os.path.isfile(index_fp):
            create_index(tfr_fp, index_fp)
        dataset = TFRecordDataset(tfr_fp, index_fp)
        data_list = list(dataset)
        for ep in range(len(data_list)):
            for step in range(bair_ep_length):
                img_data = data_list[ep][str(step) + '/image_aux1/encoded']
                # img_data = data_list[ep][str(i) + '/image_main/encoded']
                action = data_list[ep][str(step) + '/action']
                obs = np.array(img_data).reshape(1, 64, 64, 3).transpose((0, 3, 1, 2))
                D.append(obs, np.array(action), np.array(0.0), (step + 1) % bair_ep_length == 0)
    return D
'''