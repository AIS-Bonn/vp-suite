import os
import random
import numpy as np
import torch
import imageio
import torchfile
from pathlib import Path

from vp_suite.dataset._base_dataset import BaseVPDataset, VPData
import vp_suite.constants as constants

class KTHActionsDataset(BaseVPDataset):

    '''
    Some sequences might even be shorter than 30 frames;
    There, the last frame is repeated to reach MAX_SEQ_LEN.
    Going beyond 30 frames is therefore not recommended.
    Code by Angel Villar-Corrales, modified.
    '''
    NAME = "KTH Actions"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "kth_actions"
    CLASSES = ['boxing', 'handclapping', 'handwaving', 'walking', 'running', 'jogging']
    SHORT_CLASSES = ['walking', 'running', 'jogging']

    max_seq_len = 30
    action_size = 0
    frame_shape = (64, 64, 3)

    def __init__(self, split, img_processor, **dataset_kwargs):
        super(KTHActionsDataset, self).__init__(split, img_processor, **dataset_kwargs)

        self.data_dir = str((Path(self.data_dir) / "processed").resolve())
        torchfile_name = f'{self.split}_meta{self.frame_shape[0]}x{self.frame_shape[1]}.t7'
        self.data = {c: torchfile.load(os.path.join(self.data_dir, c, torchfile_name)) for c in self.CLASSES}

    def _config(self):
        return {
            "classes": self.CLASSES,
            "short_classes": self.SHORT_CLASSES
        }

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

        assert self.ready_for_usage, \
            "Dataset is not yet ready for usage (maybe you forgot to call set_seq_len())."

        c, vid, seq = self.get_from_idx(i)
        dname = os.path.join(self.data_dir, c, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.seq_len, *self.frame_shape))
        first_frame = 0 if len(seq) <= self.seq_len else random.randint(0, len(seq) - self.seq_len)
        last_frame = len(seq) - 1 if len(seq) <= self.seq_len else first_frame + self.seq_len - 1
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            frames[i - first_frame] = imageio.imread(fname)
        for i in range(last_frame + 1, self.seq_len):  # fill short sequences with repeated last frames
            frames[i] = frames[last_frame]

        rgb = self.preprocess_img(np.array(frames))  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = { "frames": rgb, "actions": actions }
        return data

    def __len__(self):
        return sum([sum([len(vid[b'files']) for vid in c_data]) for c_data in self.data.values()])

    def download_and_prepare_dataset(self):
        """
        Downloads and parepares the KTH datasets, using bash scripts from https://github.com/edenton/svg
        that have been modified by Ani Karapetyan.
        """
        from vp_suite.utils.utils import run_command
        import vp_suite.constants as constants
        run_command(f"{(constants.PKG_RESOURCES / 'download_kth.sh').resolve()} {self.DEFAULT_DATA_DIR}", print_to_console=False)
        run_command(f"{(constants.PKG_RESOURCES / 'convert_kth.sh').resolve()} {self.DEFAULT_DATA_DIR}", print_to_console=False)
