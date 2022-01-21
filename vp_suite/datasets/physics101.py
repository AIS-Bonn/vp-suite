import os
import random
import numpy as np
import torch
import imageio
import torchfile
import torchvision.transforms as TF
from torchvision.io import read_video
from pathlib import Path

from vp_suite.base.base_dataset import BaseVPDataset, VPData
import vp_suite.constants as constants


class CropUpperRight(torch.nn.Module):
    def __init__(self, w):
        super(CropUpperRight, self).__init__()
        self.w = w

    def forward(self, img):
        return img[:, :, :self.w, -self.w:]


class Phys101(BaseVPDataset):
    r"""

    """
    NAME = "Physics 101"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "phys101"
    AVAILABLE_CAMERAS = ["Camera_1", "Camera_2", "Kinect_RGB_1"]  #, "Kinect_FullDepth_1", "Kinect_RGB-D_1"]
    AVAILABLE_SUBSEQ = ["start", "middle", "end"]
    TRAIN_TO_TEST_RATIO = 0.8

    TRAIN_TEST_SEED = 1612  #: The seed to separate training data from test data; Taken from the Noether Networks code
    MIN_SEQ_LEN = 16  #: Minimum number of frames across all sequences
    ACTION_SIZE = 0  #: No actions given
    DEFAULT_FRAME_SHAPE = (1920, 1080, 3) # for cams without depth information
    DEFAULT_CAMERA = "Kinect_RGB_1"  #: Which camera to take from the dataset. TODO explanations
    DEFAULT_SUBSEQ = "middle"  #: Whether to take a sequence from the middle of the clip or the beginning

    def __init__(self, split, img_processor, **dataset_kwargs):
        r"""

        Args:
            split ():
            img_processor ():
            **dataset_kwargs ():
        """
        super(Phys101, self).__init__(split, img_processor, **dataset_kwargs)

        camera = dataset_kwargs.get("camera", self.DEFAULT_CAMERA)
        if camera not in self.AVAILABLE_CAMERAS:
            raise ValueError(f"invalid value provided for parameter '{camera}'")
        self.camera = camera

        subseq = dataset_kwargs.get("subseq", self.DEFAULT_SUBSEQ)
        if subseq not in self.AVAILABLE_SUBSEQ:
            raise ValueError(f"invalid value provided for parameter '{subseq}'")
        self.subseq = subseq

        train_test_seed = dataset_kwargs.get("train_test_seed", self.TRAIN_TEST_SEED)
        if not isinstance(train_test_seed, bool):
            raise ValueError(f"invalid value provided for parameter '{train_test_seed}'")
        self.train_test_seed = train_test_seed

        # get video filepaths for train/val or test
        self.vid_filepaths: [Path] = sorted(list(self.data_dir.rglob(f"**/{self.camera}.mp4")))
        random.Random(self.train_test_seed).shuffle(self.vid_filepaths)
        slice_idx = int(len(self.vid_filepaths) * self.TRAIN_TO_TEST_RATIO)
        if self.split == "train":
            self.vid_filepaths = self.vid_filepaths[:slice_idx]
        else:
            self.vid_filepaths = self.vid_filepaths[slice_idx:]

    def _config(self):
        raise NotImplementedError


    def __getitem__(self, i) -> VPData:
        r"""

        Args:
            i ():

        Returns:

        """
        # loaded video shape: [T, h, w, c], sitting in index 0 of the object returned by read_video()
        vid = read_video(str(self.vid_filepaths[i].resolve()), pts_unit='sec')[0]
        if self.seq_step > 1:
            vid = vid[::self.seq_step]  # [t, h, w, c]

        if self.subseq == "start":
            vid = vid[:self.total_frames]
        elif self.subseq == "end":
            vid = vid[-self.total_frames:]
        if self.subseq == 'middle':
            frame_offset = (vid.shape[0] - self.total_frames) // 2
            vid = vid[frame_offset : frame_offset+self.total_frames]

        vid = self.transformations(vid.permute(0,3,1,2)) / 255.  # [t, c, h, w], value range = [0.0, 1.0] TODO
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
        if len(vid_filepaths) == 0:  # no data available
            tar_fname = "phys101_v1.0.tar"
            tar_path = str(d_path / tar_fname)
            if not os.path.exists(tar_path):
                URL = f"http://phys101.csail.mit.edu/data/{tar_fname}"
                from vp_suite.utils.utils import download_from_url
                download_from_url(URL, tar_path)

            print("Extracting data...")
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(d_path)
            tar.close()
            os.remove(tar_path)
