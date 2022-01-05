import torch
from torch.utils.data.dataset import Dataset
from typing import TypedDict
from copy import deepcopy

class VPData(TypedDict):
    frames: torch.Tensor  # shape: [t, c, h, w]
    actions: torch.Tensor  # shape: [t, a]

class BaseVPDataset(Dataset):

    MAX_SEQ_LEN = NotImplemented
    NAME = NotImplemented
    ACTION_SIZE = NotImplemented
    DEFAULT_FRAME_SHAPE = NotImplemented
    DEFAULT_DATA_DIR = NotImplemented
    VALID_SPLITS = ["train", "test"]
    TRAIN_KEEP_RATIO = 0.8

    def __init__(self, split, **dataset_kwargs):

        super(BaseVPDataset, self).__init__()
        assert split in self.VALID_SPLITS, \
            f"parameter '{split}' has to be one of the following: {self.VALID_SPLITS}"
        self.split = split
        self.seq_step = dataset_kwargs.get("step", 1)
        self.data_dir = dataset_kwargs.get("data_dir", None)
        if self.data_dir is None:
            if self.default_available(self.split, **dataset_kwargs):
                self.data_dir = self.DEFAULT_DATA_DIR
            else:
                self.download_and_prepare_dataset()
        #self.img_processor = cfg.img_processor  # TODO don't give to dataset

    def set_seq_len(self, seq_step, context_frames, pred_frames, step=None):
        total_frames = context_frames + pred_frames
        self.total_frames = total_frames
        if step is not None:
            self.seq_step = seq_step
        self.seq_len = (self.total_frames - 1) * self.seq_step + 1
        assert self.MAX_SEQ_LEN >= self.seq_len, \
            f"Dataset '{self.NAME}' supports videos with up to {self.MAX_SEQ_LEN} frames, " \
            f"which is exceeded by your configuration: " \
            f"{{context frames: {context_frames}, pred frames: {pred_frames}, seq step: {seq_step}}}"
        self.frame_offsets = range(0, (context_frames + pred_frames) * seq_step, seq_step)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i) -> VPData:
        raise NotImplementedError

    def preprocess_img(self, img):
        return self.img_processor.preprocess_img(img)

    def default_available(self, split, **dataset_kwargs):
        """
        Tries to load a dataset and a datapoint using the default data_dir value.
        If this succeeds, then we can safely use the default data dir,
        otherwise a new dataset has to be downloaded and prepared.
        """
        try:
            kwargs_ = deepcopy(dataset_kwargs)
            kwargs_.update({"data_dir": self.DEFAULT_DATA_DIR})
            default_ = self.__class__(split, **kwargs_)
            item = default_[0]
        except Exception as e:  # TODO better handling (FileNotFound, Value, ...)
            return False
        return True

    def download_and_prepare_dataset(self):
        """
        Downloads the specific dataset, prepares it for the video prediction task (if needed)
        and stores it in a default location in the 'data/' folder.
        Implemented by the derived dataset classes
        """
        raise NotImplementedError

    @classmethod
    def get_train_val(cls, **dataset_args):
        if cls.VALID_SPLITS == ["train", "test"]:
            D_main = cls("train", **dataset_args)
            len_train = int(len(D_main) * cls.TRAIN_KEEP_RATIO)
            len_val = len(D_main) - len_train
            D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
        elif cls.VALID_SPLITS == ["train", "val", "test"]:
            D_train = cls("train", **dataset_args)
            D_val = cls("val", **dataset_args)
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
