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
    VALID_SPLITS = ["train", "test"]
    TRAIN_KEEP_RATIO = 0.8

    def __init__(self, cfg, split):

        super(BaseVPDataset, self).__init__()
        self.check_split(split)
        self.set_seq_len(cfg.data_seq_step, cfg.context_frames, cfg.pred_frames)
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
        self.frame_offsets = range(0, (context_frames + pred_frames) * seq_step, seq_step)

    def check_split(self, split):
        assert split in self.VALID_SPLITS,\
            f"parameter '{split}' has to be one of the following: {self.VALID_SPLITS}"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i) -> VPData:
        raise NotImplementedError

    def preprocess_img(self, img):
        return self.img_processor.preprocess_img(img)

    @classmethod
    def get_train_val(cls, cfg):
        if cls.VALID_SPLITS == ["train", "test"]:
            D_main = cls(cfg, "train")
            len_train = int(len(D_main) * cls.TRAIN_KEEP_RATIO)
            len_val = len(D_main) - len_train
            D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
        elif cls.VALID_SPLITS == ["train", "val", "test"]:
            D_train = cls(cfg, "train")
            D_val = cls(cfg, "val")
        else:
            raise ValueError(f"parameter 'VALID_SPLITS' of dataset class '{cls.__name__}' is ill-configured")
        return D_train, D_val

    @classmethod
    def get_test(cls, cfg):
        D_test = cls(cfg, "test")
        return D_test

    @classmethod
    def get_train_val_test(cls, cfg):
        D_train, D_val = cls.get_train_val(cfg)
        D_test = cls.get_test(cfg)
        return D_train, D_val, D_test
