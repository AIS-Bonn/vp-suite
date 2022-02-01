import json
import random
import os

import cv2
import torch
from tqdm import tqdm

from vp_suite.base.base_dataset import VPDataset, VPData
import vp_suite.constants as constants
from vp_suite.utils.utils import set_from_kwarg, read_video


class CaltechPedestrianDataset(VPDataset):
    r"""

    """
    NAME = "Caltech Pedestrian"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "caltech_pedestrian"
    VALID_SPLITS = ["train", "val", "test"]
    MIN_SEQ_LEN = 568  #: Minimum number of frames across all sequences (1322 in 2nd-shortest, 2175 in longest)
    ACTION_SIZE = 0  #: No actions given
    DATASET_FRAME_SHAPE = (480, 640, 3)
    FPS = 30  #: TODO
    TRAIN_VAL_SETS = [f"set{i:02d}" for i in range(6)]  #: TODO
    TEST_SETS = [f"set{i:02d}" for i in range(6, 11)]  #: TODO

    train_to_val_ratio = 0.9  #: big dataset -> val can be smaller
    train_val_seed = 1234  #: The seed to separate training/validation data from the previously split training data

    def __init__(self, split, **dataset_kwargs):
        r"""

        Args:
            split ():
            **dataset_kwargs ():
        """
        super(CaltechPedestrianDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["sequences", "sequences_with_frame_index",
                                     "AVAILABLE_CAMERAS"])

        # set attributes
        set_from_kwarg(self, "train_to_val_ratio", self.train_to_val_ratio, dataset_kwargs)
        set_from_kwarg(self, "train_val_seed", self.train_val_seed, dataset_kwargs)

        # get sequence filepaths and slice accordingly
        with open(os.path.join(self.data_dir, "frame_counts.json"), "r") as frame_counts_file:
            sequences = json.load(frame_counts_file).items()

        if self.split == "test":
            sequences = [(fp, frames) for (fp, frames) in sequences if fp.split("/")[-2] in self.TEST_SETS]
            if len(sequences) < 1:
                raise ValueError(f"Dataset {self.NAME}: didn't find enough test sequences "
                                 f"-> can't use dataset")
        else:
            sequences = [(fp, frames) for (fp, frames) in sequences if fp.split("/")[-2] in self.TRAIN_VAL_SETS]
            if len(sequences) < 2:
                raise ValueError(f"Dataset {self.NAME}: didn't find enough train/val sequences "
                                 f"-> can't use dataset")
            slice_idx = max(1, int(len(sequences) * self.train_to_val_ratio))
            random.Random(self.train_val_seed).shuffle(sequences)
            if self.split == "train":
                sequences = sequences[:slice_idx]
            else:
                sequences = sequences[slice_idx:]
        self.sequences = sequences

        self.sequences_with_frame_index = []  # mock value, must not be used for iteration till sequence length is set

    def _set_seq_len(self):
        r"""
        Determine per video which frame indices are valid

        Returns:

        """
        for sequence_path, frame_count in self.sequences:
            valid_start_idx = range(0, frame_count - self.seq_len + 1,
                                    self.seq_len + self.seq_step - 1)
            for idx in valid_start_idx:
                self.sequences_with_frame_index.append((sequence_path, idx))

    def __getitem__(self, i) -> VPData:
        r"""

        Args:
            i ():

        Returns:

        """
        sequence_path, start_idx = self.sequences_with_frame_index[i]
        vid = read_video(sequence_path, start_index=start_idx, num_frames=self.seq_len)  # [T, h, w, c]
        vid = vid[::self.seq_step]  # [t, h, w, c]
        vid = self.preprocess(vid)  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = {"frames": vid, "actions": actions}
        return data

    def __len__(self):
        r"""

        Returns:

        """
        return len(self.sequences_with_frame_index)

    @classmethod
    def download_and_prepare_dataset(cls):
        r"""

        Returns:

        """
        d_path = cls.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)

        # download and extract sequences if we can't find them in our folder yet
        try:
            _ = next(d_path.rglob(f"**/*.seq"))
            print(f"Found sequence data in {str(d_path.resolve())} -> Won't download {cls.NAME}")
        except StopIteration:
            from vp_suite.utils.utils import run_shell_command
            import vp_suite.constants as constants
            prep_script = (constants.PKG_RESOURCES / 'get_dataset_caltech_pedestrian.sh').resolve()
            run_shell_command(f"{prep_script} {cls.DEFAULT_DATA_DIR}")

        # pre-count frames of all sequences if not yet done so (makes data fetching faster later on)
        frame_count_path = d_path / "frame_counts.json"
        if not frame_count_path.exists():
            print(f"Analyzing video frame counts...")
            sequences = sorted(list(d_path.rglob("**/*.seq")))
            sequences_with_frame_counts = dict()
            for seq in tqdm(sequences):
                fp = str(seq.resolve())
                cap = cv2.VideoCapture(fp)
                # for these .seq files, cv2.CAP_PROP_FRAME_COUNT returns garbage,
                # so we have to manually read out the seq
                frames = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames += 1
                sequences_with_frame_counts[fp] = frames
            with open(str(frame_count_path.resolve()), "w") as frame_count_file:
                json.dump(sequences_with_frame_counts, frame_count_file)
