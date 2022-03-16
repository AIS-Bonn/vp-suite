import os
import random
import numpy as np
import torch
import imageio
import torchfile
from pathlib import Path

from vp_suite.base import VPDataset, VPData
from vp_suite.defaults import SETTINGS

class KTHActionsDataset(VPDataset):
    r"""
    Dataset class for the dataset "KTH Action", as mentioned in
    "Recognizing human actions: a local SVM approach" by Schuldt et al.
    (https://ieeexplore.ieee.org/document/1334462).

    Each sequence depicts a human acting according to one of six scenarios (actions).

    Code by Angel Villar-Corrales, modified.

    Note:
        Some sequences might even be shorter than 30 frames;
        There, the last frame is repeated to reach MAX_SEQ_LEN.
        Going beyond 30 frames is therefore not recommended.
    """
    NAME = "KTH Actions"
    REFERENCE = "https://doi.org/10.1109/ICPR.2004.1334462"
    IS_DOWNLOADABLE = "Yes"
    DEFAULT_DATA_DIR = SETTINGS.DATA_PATH / "kth_actions"
    CLASSES = ['boxing', 'handclapping', 'handwaving', 'walking', 'running', 'jogging']  #: The different scenarios that constitue this dataset.
    SHORT_CLASSES = ['walking', 'running', 'jogging']  #: Those scenarios where the sequence length might drop below the required frame count.
    MIN_SEQ_LEN = 30
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (64, 64, 3)

    first_frame_rng_seed = 1234  #: Seed value for the random number generator used to determine the first frame out of a bigger sequence.

    def __init__(self, split, **dataset_kwargs):
        super(KTHActionsDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["data"])

        self.data_dir = str((Path(self.data_dir) / "processed").resolve())
        torchfile_name = f'{self.split}_meta{self.DATASET_FRAME_SHAPE[0]}x{self.DATASET_FRAME_SHAPE[1]}.t7'
        self.data = {c: torchfile.load(os.path.join(self.data_dir, c, torchfile_name)) for c in self.CLASSES}

    def get_from_idx(self, i):
        for c, c_data in self.data.items():
            len_c_data = sum([len(vid[b'files']) for vid in c_data])
            if i >= len_c_data:  # seq sits in another class
                i -= len_c_data
                continue
            else:  # seq sits in this class
                for vid in c_data:
                    len_vid = len(vid[b'files'])
                    if i < len_vid:  # seq sits in this vid chunk
                        return c, vid, vid[b'files'][i]
                    else:  # seq sits in another vid chunk
                        i -= len_vid
        raise ValueError("invalid i")

    def __getitem__(self, i) -> VPData:
        if not self.ready_for_usage:
            raise RuntimeError("Dataset is not yet ready for usage (maybe you forgot to call set_seq_len()).")

        c, vid, seq = self.get_from_idx(i)
        dname = os.path.join(self.data_dir, c, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.seq_len, *self.DATASET_FRAME_SHAPE))
        if len(seq) <= self.seq_len:
            first_frame = 0
        else:
            first_frame = random.Random(self.first_frame_rng_seed).randint(0, len(seq) - self.seq_len)
        last_frame = len(seq) - 1 if len(seq) <= self.seq_len else first_frame + self.seq_len - 1
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            frames[i - first_frame] = imageio.imread(fname)
        for i in range(last_frame + 1, self.seq_len):  # fill short sequences with repeated last frames
            frames[i] = frames[last_frame]

        rgb = self.preprocess(np.array(frames))  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = {"frames": rgb, "actions": actions, "origin": f"{dname}, start frame: {first_frame}"}
        return data

    def __len__(self):
        return sum([sum([len(vid[b'files']) for vid in c_data]) for c_data in self.data.values()])

    @classmethod
    def download_and_prepare_dataset(cls):
        from vp_suite.utils.utils import run_shell_command
        from vp_suite.defaults import SETTINGS
        get_kth_command = f"{(SETTINGS.PKG_RESOURCES / 'get_dataset_kth.sh').resolve()} " \
                          f"{str(cls.DEFAULT_DATA_DIR.resolve())}"
        run_shell_command(get_kth_command)
