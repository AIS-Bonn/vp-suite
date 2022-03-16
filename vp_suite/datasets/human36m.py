import os
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from vp_suite.base import VPDataset, VPData
from vp_suite.defaults import SETTINGS
from vp_suite.utils.utils import set_from_kwarg, get_frame_count, read_video


class Human36MDataset(VPDataset):
    r"""
    Dataset class for the Videos of the dataset "Human 3.6M", as encountered in
    "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments"
    by Ionescu et al. (http://vision.imar.ro/human3.6m/pami-h36m.pdf).

    Each sequence depicts a human actor in a room equipped with different cameras and sensors. The actor
    is one of several different scenarios such as "Discussion", "Sitting" or "Smoking".
    """
    NAME = "Human 3.6M"
    REFERENCE = "http://vision.imar.ro/human3.6m/description.php"
    IS_DOWNLOADABLE = "With Registered Account"
    DEFAULT_DATA_DIR = SETTINGS.DATA_PATH / "human36m"
    VALID_SPLITS = ["train", "val", "test"]
    MIN_SEQ_LEN = 994  #: Minimum number of frames across all sequences (6349 in longest).
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (1000, 1000, 3) #: For Human 3.6M, some sequences come in a shape of (1002, 1000, 3). They're resized during loading.
    FPS = 50  #: Frames per Second.
    SKIP_FIRST_N = 25  #: Some of the sequences start with a bit of idling from the actor. Therefore, the first few frames of each sequence are discarded.
    ALL_SCENARIOS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
                     'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
                     'Waiting', 'WalkDog', 'WalkTogether', 'Walking', 'WalkingDog']  #: All recorded scenarios of the dataset.

    train_to_val_ratio = 0.96
    scenarios = None  #: Scenarios chosen for the current dataset instance (defaults to `self.ALL_SCENARIOS`)

    def __init__(self, split, **dataset_kwargs):
        super(Human36MDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["sequences", "sequences_with_frame_index",
                                     "ALL_SCENARIOS"])

        # set attributes
        set_from_kwarg(self, dataset_kwargs, "scenarios", default=self.ALL_SCENARIOS, choices=self.ALL_SCENARIOS)
        set_from_kwarg(self, dataset_kwargs, "train_val_seed")

        # get video filepaths for train, val or test
        split_ing = "testing" if self.split == "test" else "training"
        self.data_dir = str((Path(self.data_dir) / (split_ing)).resolve())
        with open(os.path.join(self.data_dir, "frame_counts.json"), "r") as frame_counts_file:
            self.sequences = json.load(frame_counts_file)

        # remove all videos that don't correspond to one of the selected scenarios
        self.sequences = {vfp: f for vfp, f in self.sequences.items()
                          if vfp.split("/")[-1].split(".")[0].split(" ")[0] in self.scenarios}

        # if creating training or test set, random-permute and slice
        if self.split in ["train", "val"]:
            vfc_list = list(self.sequences.items())
            slice_idx = int(len(vfc_list) * self.train_to_val_ratio)
            random.Random(self.train_val_seed).shuffle(vfc_list)
            if self.split == "train":
                self.sequences = dict(vfc_list[:slice_idx])
            else:
                self.sequences = dict(vfc_list[slice_idx:])

        self.sequences_with_frame_index = []  # mock value, must not be used for iteration till sequence length is set

    def _set_seq_len(self):
        # Determine per video which frame indices are valid
        for vfp, frame_count in self.sequences.items():
            valid_idx = range(self.SKIP_FIRST_N,
                              frame_count - self.seq_len + 1,
                              self.seq_len + self.seq_step - 1)
            for idx in valid_idx:
                self.sequences_with_frame_index.append((vfp, idx))

    def __getitem__(self, i) -> VPData:
        sequence_path, start_idx = self.sequences_with_frame_index[i]
        vid = read_video(sequence_path, img_size=self.img_shape[1:],
                         start_index=start_idx, num_frames=self.seq_len)  # [T, h, w, c]
        vid = vid[::self.seq_step]  # [t, h, w, c]
        vid = self.preprocess(vid)  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = {"frames": vid, "actions": actions, "origin": f"{sequence_path}, start frame: {start_idx}"}
        return data

    def __len__(self):
        return len(self.sequences_with_frame_index)

    @classmethod
    def download_and_prepare_dataset(cls):
        d_path = cls.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)
        vid_filepaths: [Path] = list(d_path.rglob(f"**/*.mp4"))
        if len(vid_filepaths) == 0:  # no data available -> download data and unpack
            from vp_suite.utils.utils import run_shell_command
            from vp_suite.defaults import SETTINGS
            print(f"Downloading and extracting {cls.NAME} - Videos...")
            prep_script = (SETTINGS.PKG_RESOURCES / 'get_dataset_human36m.sh').resolve()
            run_shell_command(f"{prep_script} {cls.DEFAULT_DATA_DIR}")

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
