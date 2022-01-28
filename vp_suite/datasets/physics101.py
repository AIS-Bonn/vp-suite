import os
import random
import numpy as np
import torch
import imageio
import torchfile
import torchvision.transforms as TF
from torchvision.io import read_video
from pathlib import Path

from vp_suite.base.base_dataset import VPDataset, VPData
import vp_suite.constants as constants
from vp_suite.utils.utils import set_from_kwarg, read_mp4

class CropUpperRight(torch.nn.Module):
    def __init__(self, w):
        super(CropUpperRight, self).__init__()
        self.w = w

    def forward(self, img):
        return img[:, :, :self.w, -self.w:]


class Physics101Dataset(VPDataset):
    r"""

    """
    NAME = "Physics 101"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "phys101"
    AVAILABLE_CAMERAS = ["Camera_1", "Camera_2", "Kinect_RGB_1"]  #, "Kinect_FullDepth_1", "Kinect_RGB-D_1"]
    AVAILABLE_SUBSEQ = ["start", "middle", "end"]
    MIN_SEQ_LEN = 16  #: Minimum number of frames across all sequences
    ACTION_SIZE = 0  #: No actions given
    DATASET_FRAME_SHAPE = (1080, 1920, 3) # for cams without depth information

    camera = "Kinect_RGB_1"  #: Which camera to take from the dataset. TODO explanations
    subseq = "middle"  #: Whether to take a sequence from the middle of the clip or the beginning
    trainval_to_test_ratio = 0.8
    trainval_test_seed = 1612  #: The seed to separate training data from test data; Value from the 'Noether Networks' code

    def __init__(self, split, **dataset_kwargs):
        r"""

        Args:
            split ():
            **dataset_kwargs ():
        """
        super(Physics101Dataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["AVAILABLE_CAMERAS", "AVAILABLE_SUBSEQ", "vid_filepaths"])

        # set attributes
        set_from_kwarg(self, "camera", self.camera, dataset_kwargs, choices=self.AVAILABLE_CAMERAS)
        set_from_kwarg(self, "subseq", self.subseq, dataset_kwargs, choices=self.AVAILABLE_SUBSEQ)
        set_from_kwarg(self, "train_test_seed", self.trainval_test_seed, dataset_kwargs)

        # get video filepaths for train/val or test
        self.vid_filepaths: [Path] = sorted(list(Path(self.data_dir).rglob(f"**/{self.camera}.mp4")))
        random.Random(self.trainval_test_seed).shuffle(self.vid_filepaths)
        slice_idx = int(len(self.vid_filepaths) * self.trainval_to_test_ratio)
        if self.split == "train":
            self.vid_filepaths = self.vid_filepaths[:slice_idx]
        else:
            self.vid_filepaths = self.vid_filepaths[slice_idx:]

    def __getitem__(self, i) -> VPData:
        r"""

        Args:
            i ():

        Returns:

        """
        # loaded video shape: [T, h, w, c], sitting in index 0 of the object returned by read_video()
        vid = read_mp4(self.vid_filepaths[i])  # [T, h, w, c]
        if self.seq_step > 1:
            vid = vid[::self.seq_step]  # [t, h, w, c]

        if self.subseq == "start":
            vid = vid[:self.total_frames]
        elif self.subseq == "end":
            vid = vid[-self.total_frames:]
        if self.subseq == 'middle':
            frame_offset = (vid.shape[0] - self.total_frames) // 2
            vid = vid[frame_offset : frame_offset+self.total_frames]

        vid = self.preprocess(vid)
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = { "frames": vid, "actions": actions }
        return data

    def __len__(self):
        r"""

        Returns:

        """
        return len(self.vid_filepaths)

    def download_and_prepare_dataset(self):
        r"""

        Returns:

        """
        d_path = self.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)
        vid_filepaths: [Path] = list(d_path.rglob(f"**/*.mp4"))
        if len(vid_filepaths) == 0:  # no data available -> unpack tar
            tar_fname = "phys101_v1.0.tar"
            tar_path = str(d_path / tar_fname)
            if not os.path.exists(tar_path):  # no tar available -> download it
                URL = f"http://phys101.csail.mit.edu/data/{tar_fname}"
                from vp_suite.utils.utils import download_from_url
                download_from_url(URL, tar_path)

            print("Extracting data...")
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(d_path)
            tar.close()
            os.remove(tar_path)
