import torch
from torch.utils.data.dataset import Dataset
from typing import TypedDict
from copy import deepcopy
from pathlib import Path

from vp_suite.utils.img_processor import ImgProcessor

class VPData(TypedDict):
    r"""TODO

    """
    frames: torch.Tensor  #: shape: [t, c, h, w]
    actions: torch.Tensor  #: shape: [t, a]

class BaseVPDataset(Dataset):
    r"""

    Attributes:

    """

    NAME: str = NotImplemented  #: the dataset's name.
    DEFAULT_DATA_DIR: Path = NotImplemented  #: the default save location of the dataset files.
    VALID_SPLITS = ["train", "test"]  #: the valid arguments for specifying splits.
    MIN_SEQ_LEN: int = NotImplemented  #: TODO
    ACTION_SIZE: int = NotImplemented  #: TODO
    DEFAULT_FRAME_SHAPE: (int, int, int) = NotImplemented  #: TODO
    train_keep_ratio: float = 0.8  #: TODO

    def __init__(self, split, img_processor, **dataset_kwargs):
        r"""

        Args:
            split ():
            img_processor ():
            **dataset_kwargs ():
        """

        super(BaseVPDataset, self).__init__()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"parameter '{split}' has to be one of the following: {self.VALID_SPLITS}")
        self.split = split
        self.seq_step = dataset_kwargs.get("seq_step", 1)
        self.data_dir : Path = dataset_kwargs.get("data_dir", None)
        if self.data_dir is None:
            if not self.default_available(self.split, img_processor, **dataset_kwargs):
                print(f"downloading/preparing dataset '{self.NAME}' "
                      f"and saving it to '{self.DEFAULT_DATA_DIR}'...")
                self.download_and_prepare_dataset()
            self.data_dir = self.DEFAULT_DATA_DIR
        self.img_processor : ImgProcessor = img_processor
        self.ready_for_usage = False  # becomes True once sequence length has been set

    @property
    def config(self):
        r"""TODO

        Returns:

        """
        img_h, img_w, img_c = self.DEFAULT_FRAME_SHAPE
        base_config = {
            "action_size": self.ACTION_SIZE,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "img_shape": self.DEFAULT_FRAME_SHAPE,
            "tensor_value_range": [self.img_processor.value_min, self.img_processor.value_max],
            "supports_actions": self.ACTION_SIZE > 0,
            "max_seq_len": self.MIN_SEQ_LEN,
            "frame_shape": self.DEFAULT_FRAME_SHAPE,
            "train_keep_ratio": self.train_keep_ratio
        }
        return {**base_config, **self._config()}

    def _config(self):
        r"""Dataset-specific config
        """
        return {}

    def set_seq_len(self, context_frames : int, pred_frames : int, seq_step : int):
        r"""
        Set the sequence length for the upcoming run. Asserts that the given parameters
        lead to a sequence length that does not exceed the possible range.

        Args:
            context_frames ():
            pred_frames ():
            seq_step ():
        """
        total_frames = context_frames + pred_frames
        seq_len = (total_frames - 1) * seq_step + 1
        if self.MIN_SEQ_LEN < seq_len:
            raise ValueError(f"Dataset '{self.NAME}' supports videos with up to {self.MIN_SEQ_LEN} frames, "
                             f"which is exceeded by your configuration: "
                             f"{{context frames: {context_frames}, pred frames: {pred_frames}, seq step: {seq_step}}}")
        self.total_frames = total_frames
        self.seq_len = seq_len
        self.frame_offsets = range(0, (total_frames) * self.seq_step, self.seq_step)
        self._set_seq_len()
        self.ready_for_usage = True

    def _set_seq_len(self):
        r"""Optional logic for datasets with specific logic

        """
        pass

    def __len__(self) -> int:
        r"""

        Returns:

        """
        raise NotImplementedError

    def __getitem__(self, i) -> VPData:
        r"""

        Args:
            i ():

        Returns:

        """
        raise NotImplementedError

    def preprocess_img(self, img):
        r"""

        Args:
            img ():

        Returns:

        """
        return self.img_processor.preprocess_img(img)

    def postprocess_img(self, img):
        r"""

        Args:
            img ():

        Returns:

        """
        return self.img_processor.postprocess_img(img)

    def default_available(self, split, img_processor, **dataset_kwargs):
        r"""
        Tries to load a dataset and a datapoint using the default data_dir value.
        If this succeeds, then we can safely use the default data dir,
        otherwise a new dataset has to be downloaded and prepared.

        Args:
            split ():
            img_processor ():
            **dataset_kwargs ():

        Returns:

        """
        try:
            kwargs_ = deepcopy(dataset_kwargs)
            kwargs_.update({"data_dir": self.DEFAULT_DATA_DIR})
            default_ = self.__class__(split, img_processor, **kwargs_)
            default_.set_seq_len(1, 1, 1)
            _ = default_[0]
        except (FileNotFoundError, ValueError):  # TODO other exceptions?
            return False
        return True

    def download_and_prepare_dataset(self):
        r"""
        Downloads the specific dataset, prepares it for the video prediction task (if needed)
        and stores it in a default location in the 'data/' folder.
        Implemented by the derived dataset classes
        """
        raise NotImplementedError

    @classmethod
    def get_train_val(cls, img_processor, **dataset_args):
        r"""

        Args:
            img_processor ():
            **dataset_args ():

        Returns:

        """
        if cls.VALID_SPLITS == ["train", "test"]:
            D_main = cls("train", img_processor, **dataset_args)
            len_train = int(len(D_main) * cls.train_keep_ratio)
            len_val = len(D_main) - len_train
            D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
        elif cls.VALID_SPLITS == ["train", "val", "test"]:
            D_train = cls("train", img_processor, **dataset_args)
            D_val = cls("val", img_processor, **dataset_args)
        else:
            raise ValueError(f"parameter 'VALID_SPLITS' of dataset class '{cls.__name__}' is ill-configured")
        return D_train, D_val

    @classmethod
    def get_test(cls, img_processor, **dataset_args):
        r"""

        Args:
            img_processor ():
            **dataset_args ():

        Returns:

        """
        D_test = cls("test", img_processor, **dataset_args)
        return D_test

    @classmethod
    def get_train_val_test(cls, img_processor, **dataset_args):
        r"""

        Args:
            img_processor ():
            **dataset_args ():

        Returns:

        """
        D_train, D_val = cls.get_train_val(img_processor, **dataset_args)
        D_test = cls.get_test(img_processor, **dataset_args)
        return D_train, D_val, D_test
