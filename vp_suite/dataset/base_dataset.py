import torch
from torch.utils.data.dataset import Dataset
from typing import TypedDict

class VPData(TypedDict):
    frames: torch.Tensor  # shape: [t, c, h, w]
    actions: torch.Tensor  # shape: [t, a]

class BaseVPDataset(Dataset):

    MAX_SEQ_LEN = NotImplemented
    NAME = NotImplemented
    ACTION_SIZE = NotImplemented
    DEFAULT_FRAME_SHAPE = NotImplemented

    def __init__(self, data_dir, cfg):

        super(BaseVPDataset, self).__init__()
        self.set_seq_len(cfg.data_seq_step, cfg.context_frames, cfg.pred_frames)
        self.data_dir = data_dir
        self.dataset_cfg = cfg
        self.img_processor = cfg.img_processor
        pass

    def set_seq_len(self, seq_step, context_frames, pred_frames):
        total_frames = context_frames + pred_frames
        self.total_frames = total_frames
        self.seq_step = seq_step
        self.seq_len = (self.total_frames - 1) * self.seq_step + 1
        assert self.MAX_SEQ_LEN >= self.seq_len, \
            f"Dataset '{self.NAME}' supports videos with up to {self.MAX_SEQ_LEN} frames, " \
            f"which is exceeded by your configuration: " \
            f"{{context frames: {context_frames}, pred frames: {pred_frames}, seq step: {seq_step}}}"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i) -> VPData:
        raise NotImplementedError

    def preprocess_img(self, img):
        return self.img_processor.preprocess_img(img)