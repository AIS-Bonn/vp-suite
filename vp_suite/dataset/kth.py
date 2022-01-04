import os
import random
import numpy as np
import torch
import imageio
import torchfile
from pathlib import Path
from vp_suite.dataset._base_dataset import BaseVPDataset, VPData

class KTHActionsDataset(BaseVPDataset):

    '''
    Some sequences might even be shorter than 30 frames;
    There, the last frame is repeated to reach MAX_SEQ_LEN.
    Going beyond 30 frames is therefore not recommended.
    '''
    MAX_SEQ_LEN = 30
    NAME = "KTH Actions"
    ACTION_SIZE = 0
    DEFAULT_FRAME_SHAPE = (64, 64, 3)
    CLASSES = ['boxing', 'handclapping', 'handwaving', 'walking', 'running', 'jogging']
    SHORT_CLASSES = ['walking', 'running', 'jogging']

    def __init__(self, cfg, split):
        super(KTHActionsDataset, self).__init__(cfg, split)

        self.split = split
        self.data_dir = str(Path(cfg.data_dir) / "processed")
        torchfile_name = f'{self.split}_meta{self.DEFAULT_FRAME_SHAPE[0]}x{self.DEFAULT_FRAME_SHAPE[1]}.t7'
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

        c, vid, seq = self.get_from_idx(i)
        dname = os.path.join(self.data_dir, c, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.seq_len, *self.DEFAULT_FRAME_SHAPE))
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

# === KTH data preparation tools ===============================================
