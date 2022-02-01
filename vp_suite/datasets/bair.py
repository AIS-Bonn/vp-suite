import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from tfrecord.tools.tfrecord2idx import create_index
from tfrecord.torch.dataset import TFRecordDataset

from vp_suite.base.base_dataset import VPDataset, VPData
import vp_suite.constants as constants

class BAIRPushingDataset(VPDataset):
    r"""

    """
    NAME = "BAIR robot pushing"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "bair_robot_pushing"
    MIN_SEQ_LEN = 30  #: a trajectory in the BAIR robot pushing dataset is 30 timesteps
    ACTION_SIZE = 4
    DATASET_FRAME_SHAPE = (64, 64, 3)

    train_to_val_ratio = 0.96  #: big dataset -> val can be smaller

    def __init__(self, split, **dataset_kwargs):
        r"""

        Args:
            split ():
            **dataset_kwargs ():
        """
        super(BAIRPushingDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["obs_ids", "actions_ids", "obs_fps", "actions_fps"])

        self.data_dir = str((Path(self.data_dir) / "softmotion30_44k" / split).resolve())
        self.obs_ids = [fn for fn in sorted(os.listdir(self.data_dir)) if str(fn).endswith("obs.npy")]
        self.actions_ids = [fn for fn in sorted(os.listdir(self.data_dir)) if str(fn).endswith("actions.npy")]

        if len(self.obs_ids) != len(self.actions_ids):
            raise ValueError("Different number of obs and action files found -> Delete dataset and prepare again!")
        elif len(self.obs_ids) == 0:
            raise ValueError("No trajectory files (.npy) found! Maybe you forgot to prepare the dataset?")

        self.obs_fps = [os.path.join(self.data_dir, i) for i in self.obs_ids]
        self.actions_fps = [os.path.join(self.data_dir, i) for i in self.actions_ids]

    def __len__(self):
        r"""

        Returns:

        """
        return len(self.obs_fps)

    def __getitem__(self, i) -> VPData:
        r"""

        Args:
            i ():

        Returns:

        """
        if not self.ready_for_usage:
            raise RuntimeError("Dataset is not yet ready for usage (maybe you forgot to call set_seq_len()).")

        rgb_raw = np.load(self.obs_fps[i])
        rgb_raw = rgb_raw[:self.seq_len:self.seq_step]
        rgb = self.preprocess(rgb_raw)  # [t, c, h, w]

        actions = torch.from_numpy(np.load(self.actions_fps[i])).float()
        actions = actions[:self.seq_len:self.seq_step]  # [t, a]

        data = { "frames": rgb, "actions": actions }
        return data

    @classmethod
    def download_and_prepare_dataset(cls):
        r"""

        Returns:

        """

        d_path = cls.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)
        ds_path = d_path / "softmotion30_44k"
        if not os.path.exists(str(ds_path)):
            download_and_extract_bair(d_path)
        print("splitting trajectory files...")
        split_bair_traj_files(ds_path / "train", True)
        split_bair_traj_files(ds_path / "test", True)


# === BAIR data preparation tools ==============================================

def download_and_extract_bair(d_path):
    r"""

    Args:
        d_path ():

    Returns:

    """
    tar_fname = "bair_robot_pushing_dataset_v0.tar"
    tar_path = str(d_path / tar_fname)
    if not os.path.exists(tar_path):
        URL = f"http://rail.eecs.berkeley.edu/datasets/{tar_fname}"
        from vp_suite.utils.utils import download_from_url
        download_from_url(URL, tar_path)

    print("Extracting data...")
    import tarfile
    tar = tarfile.open(tar_path)
    tar.extractall(d_path)
    tar.close()
    os.remove(tar_path)

def split_bair_traj_files(data_dir, delete_tfrecords):
    r"""

    Args:
        data_dir ():
        delete_tfrecords ():

    Returns:

    """
    bair_ep_length = 30

    data_files = [fn for fn in sorted(os.listdir(str(data_dir.resolve()))) if str(fn).endswith(".tfrecords")]
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
            out_obs_fp = data_dir / f"seq_{ep_number:05d}_obs.npy"
            np.save(out_obs_fp, observations)

            actions = np.concatenate(actions, axis=0)
            out_actions_fp = data_dir / f"seq_{ep_number:05d}_actions.npy"
            np.save(out_actions_fp, actions)

            ep_number += 1

        if delete_tfrecords:
            os.remove(tfr_fp)
            os.remove(index_fp)
