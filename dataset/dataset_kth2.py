import os
import json
import random
import numpy as np
import torch
import imageio
import torchfile
from torch.utils.data.dataset import Dataset
from dataset.dataset_utils import preprocess_img


class KTHActionsDataset2(Dataset):

    def __init__(self, data_path, split):
        self.data_root = os.path.join(data_path, "processed")
        self.split = split
        assert self.split in ['train', 'test']
        self.img_size = 64
        self.img_shape = (self.img_size, self.img_size)

        self.classes = ['boxing', 'handclapping', 'handwaving', 'walking', 'running', 'jogging']
        self.short_classes = ['walking', 'running', 'jogging']
        self.seq_length = 30
        self.channels = 3
        self.action_size = 0

        dataset = "train" if self.split == "train" else "test"
        self.data = {}
        for c in self.classes:
            self.data[c] = torchfile.load(os.path.join(self.data_root, c,
                                                       f'{dataset}_meta{self.img_size}x{self.img_size}.t7'))

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

    def __getitem__(self, i):

        c, vid, seq = self.get_from_idx(i)
        dname = os.path.join(self.data_root, c, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.seq_length, self.img_size, self.img_size, 3))
        first_frame = 0 if len(seq) <= self.seq_length else random.randint(0, len(seq) - self.seq_length)
        last_frame = len(seq) - 1 if len(seq) <= self.seq_length else first_frame + self.seq_length - 1
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            frames[i - first_frame] = imageio.imread(fname)

        for i in range(last_frame + 1, self.seq_length):
            frames[i] = frames[last_frame]

        rgb = preprocess_img(np.array(frames))  # [t, c, h, w]

        data = {
            "rgb": rgb,
            "actions": torch.zeros((rgb.shape[0], 1))  # actions should be disregarded in training logic
        }
        return data

    def __len__(self):
        return sum([sum([len(vid[b'files']) for vid in c_data]) for c_data in self.data.values()])