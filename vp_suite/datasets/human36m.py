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


class Human36MDataset(VPDataset):
    r"""

    """
    NAME = "Human 3.6M"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "human36m"
    TRAIN_TO_TEST_RATIO = 0.8
    MIN_SEQ_LEN = 16  #: Minimum number of frames across all sequences
    ACTION_SIZE = 0  #: No actions given
    DATASET_FRAME_SHAPE = (1920, 1080, 3) # for cams without depth information
    DEFAULT_CAMERA = "Kinect_RGB_1"  #: Which camera to take from the dataset. TODO explanations
    DEFAULT_SUBSEQ = "middle"  #: Whether to take a sequence from the middle of the clip or the beginning

    def __init__(self, split, **dataset_kwargs):
        r"""

        Args:
            split ():
            **dataset_kwargs ():
        """
        super(Human36MDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["vid_filepaths"])

        # set attributes
        set_from_kwarg(self, "camera", self.DEFAULT_CAMERA, dataset_kwargs, choices=self.AVAILABLE_CAMERAS)
        set_from_kwarg(self, "subseq", self.DEFAULT_SUBSEQ, dataset_kwargs, choices=self.AVAILABLE_SUBSEQ)
        set_from_kwarg(self, "train_test_seed", self.TRAIN_TEST_SEED, dataset_kwargs)

        # get video filepaths for train/val or test
        self.vid_filepaths: [Path] = sorted(list(Path(self.data_dir).rglob(f"**/{self.camera}.mp4")))
        random.Random(self.train_test_seed).shuffle(self.vid_filepaths)
        slice_idx = int(len(self.vid_filepaths) * self.TRAIN_TO_TEST_RATIO)
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

    def _prepare_human36m(data_dir, out_dir):
        person_dirs = sorted(os.listdir(os.path.join(data_dir, "subject")))
        person_dirs = [os.path.join(data_dir, "subject", pd, "Videos") for pd in person_dirs]
        files = sorted([os.path.join(pd, fn) for pd in person_dirs
                        for fn in os.listdir(pd) if str(fn).endswith(".mp4")])
        for fp in tqdm(files):
            relative_fp = Path(fp).relative_to(Path(data_dir))
            out_path = Path(out_dir) / relative_fp
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with VideoFileClip(fp, audio=False, target_resolution=(64, 64), resize_algorithm="bilinear") as clip:
                clip.write_videofile(os.path.join(out_dir, relative_fp), audio=False, codec="mpeg4", logger=None)

    def prepare_human36m(data_dir, out_dir):
        _prepare_human36m(os.path.join(data_dir, "training"), os.path.join(out_dir, "training"))
        _prepare_human36m(os.path.join(data_dir, "testing"), os.path.join(out_dir, "testing"))