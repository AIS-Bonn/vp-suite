import os
import json
import random
import numpy as np
import torch
import imageio
import torchfile
from tqdm import tqdm
import torchvision.transforms as TF
from torchvision.io import read_video
from pathlib import Path

from vp_suite.base.base_dataset import VPDataset, VPData
import vp_suite.constants as constants
from vp_suite.utils.utils import set_from_kwarg, get_frame_count, read_mp4


class Human36MDataset(VPDataset):
    r"""

    """
    NAME = "Human 3.6M"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "human36m"
    VALID_SPLITS = ["train", "val", "test"]
    MIN_SEQ_LEN = 994  #: Minimum number of frames across all sequences (6349 in longest)
    ACTION_SIZE = 0  #: No actions given
    DATASET_FRAME_SHAPE = (1000, 1000, 3) # some videos have (1002, 1000, 3) -> resize during loading
    FPS = 50
    SKIP_FIRST_N = 25  # some of the sequence start with a tiny bit of idling
    ALL_SCENARIOS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
                     'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
                     'Waiting', 'WalkDog', 'WalkTogether', 'Walking', 'WalkingDog']

    train_keep_ratio = 0.96  #: big dataset -> val can be smaller
    train_val_seed = 1234
    scenarios = None

    def __init__(self, split, **dataset_kwargs):
        r"""

        Args:
            split ():
            **dataset_kwargs ():
        """
        super(Human36MDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["vid_frame_counts", "vid_filepaths_with_frame_index",
                                     "ALL_SCENARIOS"])

        # set attributes
        set_from_kwarg(self, "scenarios", self.ALL_SCENARIOS, dataset_kwargs, choices=self.ALL_SCENARIOS)
        set_from_kwarg(self, "train_val_seed", self.train_val_seed, dataset_kwargs)

        # get video filepaths for train, val or test
        split_ing = "testing" if split == "test" else "training"
        self.data_dir = str((Path(self.data_dir) / (split_ing)).resolve())
        with open(os.path.join(self.data_dir, "frame_counts.json"), "r") as frame_counts_file:
            self.vid_frame_counts = json.load(frame_counts_file)

        # remove all videos that don't correspond to one of the selected scenarios
        self.vid_frame_counts = {vfp: f for vfp, f in self.vid_frame_counts.items()
                                 if vfp.split("/")[-1].split(".")[0].split(" ")[0] in self.scenarios}

        # if creating training or test set, random-permute and slice
        if split in ["train", "val"]:
            vfc_list = list(self.vid_frame_counts.items())
            slice_idx = int(len(vfc_list) * self.train_keep_ratio)
            random.Random(self.train_val_seed).shuffle(vfc_list)
            if split == "train":
                self.vid_frame_counts = dict(vfc_list[:slice_idx])
            else:
                self.vid_frame_counts = dict(vfc_list[slice_idx:])

        self.vid_filepaths_with_frame_index = []  # mock value, must not be used for iteration till sequence length is set

    def _set_seq_len(self):
        r"""
        Determine per video which frame indices are valid

        Returns:

        """
        for vfp, frame_count in self.vid_frame_counts.items():
            valid_idx = range(self.SKIP_FIRST_N,
                              frame_count - self.seq_len + 1,
                              self.seq_len + self.seq_step - 1)
            for idx in valid_idx:
                self.vid_filepaths_with_frame_index.append((vfp, idx))

    def __getitem__(self, i) -> VPData:
        r"""

        Args:
            i ():

        Returns:

        """
        # loaded video shape: [T, h, w, c], sitting in index 0 of the object returned by read_video()
        vid_fp, start_idx = self.vid_filepaths_with_frame_index[i]
        vid = read_mp4(vid_fp, img_size=self.img_shape[1:],
                       start_index=start_idx, num_frames=self.seq_len)  # [T, h, w, c]
        vid = vid[::self.seq_step]  # [t, h, w, c]
        vid = self.preprocess(vid)
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = { "frames": vid, "actions": actions }
        return data

    def __len__(self):
        r"""

        Returns:

        """
        return len(self.vid_filepaths_with_frame_index)

    def download_and_prepare_dataset(self):
        r"""

        Returns:

        """
        d_path = self.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)
        vid_filepaths: [Path] = list(d_path.rglob(f"**/*.mp4"))
        if len(vid_filepaths) == 0:  # no data available -> download data and unpack
            from vp_suite.utils.utils import run_command
            import vp_suite.constants as constants
            print("Downloading and extracting Human 3.6M - Videos...")
            prep_script = (constants.PKG_RESOURCES / 'get_dataset_human36m.sh').resolve()
            run_command(f"{prep_script} {self.DEFAULT_DATA_DIR}", print_to_console=True)

        # open all videos to get their frame counts (speeds up dataset creation later)
        print(f"Analyzing video frame counts...")
        for split in ["training", "testing"]:
            frame_counts_dict = dict()
            d_split_path = d_path / split
            frame_counts_fp = str((d_split_path / "frame_counts.json").resolve())
            vid_filepaths = list(d_split_path.rglob(f"**/*.mp4"))
            for vid_fp in tqdm(vid_filepaths, postfix=split):
                frame_count = get_frame_count(vid_fp)
                frame_counts_dict[str(vid_fp.resolve())] = frame_count
            with open(frame_counts_fp, "w") as frame_counts_file:
                json.dump(frame_counts_dict, frame_counts_file)
