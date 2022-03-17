import sys
from typing import TypedDict, Union, Sequence, List
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TF
from torch._utils import _accumulate
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset

from vp_suite.utils.utils import set_from_kwarg, get_public_attrs, PytestExpectedException


CROPS = [TF.CenterCrop, TF.RandomCrop]
SHAPE_PRESERVING_AUGMENTATIONS = [
    TF.RandomErasing, TF.Normalize, TF.RandomEqualize, TF.RandomAutocontrast, TF.RandomAdjustSharpness,
    TF.RandomSolarize, TF.RandomPosterize, TF.RandomInvert, TF.GaussianBlur, TF.RandomVerticalFlip,
    TF.RandomRotation, TF.RandomHorizontalFlip, TF.RandomGrayscale, TF.Grayscale, TF.ColorJitter,
]


class VPData(TypedDict):
    r"""
    This template class defines the return type for all datasets.
    """
    frames: torch.Tensor  #: Video frames: torch tensors of shape [t, c, h, w].
    actions: torch.Tensor  #: Actions per frame: torch tensors of shape [t, a].
    origin: str  #: A string specifying the source of the data.


class VPSubset(Subset):
    r"""
    A minimal wrapper around :class:`~Subset` that allows to directly access the underlying dataset's attributes.
    """
    def __getattr__(self, item):
        return getattr(self.dataset, item)


class VPDataset(Dataset):
    r"""
    The base class for all video prediction dataset loaders.
    Data points are provided in the shape of :class:`VPData` dicts.

    Note:
        VPDataset objects are not usable directly after creation since the sequence length is unspecified.
        In order to fully prepare the dataset, :meth:`self.set_seq_len()` has to be called with the desired amount
        of frames and the seq_step. Afterwards, the VPDataset object. is ready to be queried for data.
    """
    NON_CONFIG_VARS = ["functions",  "ready_for_usage", "total_frames", "seq_len", "frame_offsets", "data_dir"]  #: Variables that do not get included in the dict returned by :meth:`self.config()` (Constants are not included either).

    # DATASET CONSTANTS
    NAME: str = NotImplemented  #: The dataset's name.
    REFERENCE: str = None  #: The reference (publication) where the original dataset is introduced.
    IS_DOWNLOADABLE: str = None  #: A string identifying whether the dataset can be (freely) downloaded.
    ON_THE_FLY: bool = False  #: If true, accessing the dataset means data is generated on the fly rather than fetched from storage.
    DEFAULT_DATA_DIR: Path = NotImplemented  #: The default save location of the dataset files.
    VALID_SPLITS = ["train", "test"]  #: The valid arguments for specifying splits.
    MIN_SEQ_LEN: int = NotImplemented  #: The minimum sequence length provided by the dataset.
    ACTION_SIZE: int = NotImplemented  #: The size of the action vector per frame (If the dataset provides no actions, this value is 0).
    DATASET_FRAME_SHAPE: (int, int, int) = NotImplemented  #: Shape of a single frame in the dataset (height, width, channels).

    # dataset hyper-parameters
    img_shape: (int, int, int) = NotImplemented  #: Shape of a single frame as returned by `__getitem()__`.
    train_to_val_ratio: float = 0.8  #: The ratio of files that will be training data (rest will be validation data). For bigger datasets, this ratio can be set closer to 1.
    train_val_seed = 1234  #: Random seed used to separate training and validation data.
    transform: nn.Module = None  #: This module gets called in the preprocessing step and consists of pre-specified cropping, resizing and augmentation layers.
    split: str = None  #: The dataset's split identifier (i.e. whether it's a training/validation/test dataset).
    seq_step: int = 1  #: With a step N, every Nth frame is included in the returned sequence.
    data_dir: str = None  #: The specified path to the folder containing the dataset.
    value_range_min: float = 0.0  #: The lower end of the value range for the returned data.
    value_range_max: float = 1.0  #: The upper end of the value range for the returned data.

    def __init__(self, split: str, **dataset_kwargs):
        r"""
        Initializes the dataset loader by determining its split and extracting and processing
        all dataset attributes from the parameters given in `dataset_kwargs`.

        Args:
            split (str): The dataset's split identifier (i.e. whether it's a training/validation/test dataset)
            **dataset_kwargs (Any): Optional dataset arguments for image transformation, value_range, splitting etc.
        """

        super(VPDataset, self).__init__()

        if split not in self.VALID_SPLITS:
            raise ValueError(f"parameter '{split}' has to be one of the following: {self.VALID_SPLITS}")
        self.split = split

        set_from_kwarg(self, dataset_kwargs, "seq_step")
        self.data_dir = dataset_kwargs.get("data_dir", self.data_dir)
        if self.data_dir is None:
            if not self.default_available(self.split, **dataset_kwargs):
                if "pytest" in sys.modules:  # don't download datasets if running this code from the test suite
                    raise PytestExpectedException(f"Default for Dataset '{self.NAME}' is unavailable "
                                                  f"and pytest won't download it")
                else:
                    print(f"downloading/preparing dataset '{self.NAME}' "
                          f"and saving it to '{self.DEFAULT_DATA_DIR}'...")
                    self.download_and_prepare_dataset()
            self.data_dir = str(self.DEFAULT_DATA_DIR.resolve())

        # DATA PREPROCESSING: convert -> permute -> scale -> crop -> resize -> augment
        transforms = []

        # scale
        set_from_kwarg(self, dataset_kwargs, "value_range_min")
        set_from_kwarg(self, dataset_kwargs, "value_range_max")

        # crop
        crop = dataset_kwargs.get("crop", None)
        if crop is not None:
            if type(crop) not in CROPS:
                raise ValueError(f"for the parameter 'crop', only the following transforms are allowed: {CROPS}")
            transforms.append(crop)

        # resize (also sets img_shape)
        img_size = dataset_kwargs.get("img_size", None)
        h, w, c = self.DATASET_FRAME_SHAPE
        if img_size is None:
            h_, w_ = h, w
        elif isinstance(img_size, int):
            h_, w_ = img_size, img_size
        elif (isinstance(img_size, list) or isinstance(img_size, tuple)) and len(img_size) == 2:
            h_, w_ = img_size
        else:
            raise ValueError(f"invalid img size provided, expected either None, int or a two-element list/tuple")
        self.img_shape = c, h_, w_
        if h != self.img_shape[1] or w != self.img_shape[2]:
            transforms.append(TF.Resize(size=self.img_shape[1:]))

        # augment
        augmentations = dataset_kwargs.get("augmentations", [])
        for aug in augmentations:
            if type(aug) not in SHAPE_PRESERVING_AUGMENTATIONS:
                raise ValueError(f"within the parameter 'augmentations', "
                                 f"only the following transformations are allowed: {SHAPE_PRESERVING_AUGMENTATIONS}")
            transforms.append(aug)

        # FINALIZE
        self.transform = nn.Identity() if len(transforms) == 0 else nn.Sequential(*transforms)
        self.ready_for_usage = False  # becomes True once sequence length has been set

    @property
    def config(self) -> dict:
        r"""
        Returns: A dictionary containing the complete dataset configuration, including common attributes
        as well as dataset-specific attributes.
        """
        attr_dict = get_public_attrs(self, "config", non_config_vars=self.NON_CONFIG_VARS)
        img_c, img_h, img_w = self.img_shape
        extra_config = {
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "action_size": self.ACTION_SIZE,
            "tensor_value_range": [self.value_range_min, self.value_range_max],
            "NAME": self.NAME
        }
        return {**attr_dict, **extra_config}

    def set_seq_len(self, context_frames: int, pred_frames: int, seq_step: int):
        r"""
        Set the sequence length for the upcoming run. Assumes that the given parameters
        lead to a sequence length that does not exceed the minimum sequence length
        specified in :attr:`self.MIN_SEQ_LEN`.

        Args:
            context_frames (int): Number of input/context frames.
            pred_frames (int): Number of frames to be predicted.
            seq_step (int): Sequence step (for step N, assemble the sequence by taking every Nth frame).
        """
        total_frames = context_frames + pred_frames
        seq_len = (total_frames - 1) * seq_step + 1
        if self.MIN_SEQ_LEN < seq_len:
            raise ValueError(f"Dataset '{self.NAME}' supports videos with up to {self.MIN_SEQ_LEN} frames, "
                             f"which is exceeded by your configuration: "
                             f"{{context frames: {context_frames}, pred frames: {pred_frames}, seq step: {seq_step}}}")
        self.total_frames = total_frames
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.frame_offsets = range(0, (total_frames) * seq_step, seq_step)
        self._set_seq_len()
        self.ready_for_usage = True

    def _set_seq_len(self):
        r""" Optional dataset-specific logic for :meth:`self.set_seq_len()`. """
        pass

    def reset_rng(self):
        r""" Optional logic for resetting the RNG of a dataset. """
        pass

    def __len__(self) -> int:
        r"""
        Returns: The number of available data points of this dataset (its "size").

        Note: Prior to setting the sequence length, this parameter is unusable!
        """
        raise NotImplementedError

    def __getitem__(self, i) -> VPData:
        raise NotImplementedError

    def preprocess(self, x: Union[np.ndarray, torch.Tensor], transform: bool = True) -> torch.Tensor:
        r"""
        Preprocesses the input sequence to make it usable by the video prediction models.
        Makes use of the transformations defined in :meth:`self.__init__()`.
        Workflow is as follows:

        1. Convert to torch tensor of type torch.float.

        2. Permute axes to obtain the following shape: [frames/time (t), channels (c), height (h), width (w)].

        3. Scale values to the interval defined by :attr:`self.value_range_min` and :attr:`self.value_range_max`.

        4. Crop the image (if applicable).

        5. Resize the image (if applicable).

        6. Perform further data augmentation operations (if applicable).

        Args:
            x (Union[np.ndarray, torch.Tensor]): The input sequence.
            transform (bool): Whether to crop/resize/augment the sequence using the dataset's transformations.

        Returns: The preprocessed sequence tensor.
        """

        # conversion to torch float of range [0.0, 1.0]
        if isinstance(x, np.ndarray):
            if x.dtype == np.uint16:
                x = x.astype(np.float32) / ((1 << 16) - 1)
            elif x.dtype == np.uint8:
                x = x.astype(np.float32) / ((1 << 8) - 1)
            elif x.dtype == float:
                x = x / ((1 << 8) - 1)
            else:
                raise ValueError(f"if providing numpy arrays, only dtypes "
                                 f"np.uint8, np.float and np.uint16 are supported (given: {x.dtype})")
            x = torch.from_numpy(x).float()
        elif torch.is_tensor(x):
            if x.dtype == torch.uint8:
                x = x.float() / ((1 << 8) - 1)
            elif x.dtype == torch.double:
                x = x.float()
            else:
                raise ValueError(f"if providing pytorch tensors, only dtypes "
                                 f"torch.uint8, torch.float and torch.double are supported (given: {x.dtype})")
        else:
            raise ValueError(f"expected input to be either a numpy array or a PyTorch tensor")

        # assuming shape = [..., h, w(, c)], putting channel dim at index -3
        if x.ndim < 2:
            raise ValueError(f"expected at least two dimensions for input image")
        elif x.ndim == 2:
            x = x.unsqueeze(dim=0)
        else:
            permutation = list(range(x.ndim - 3)) + [-1, -3, -2]
            x = x.permute(permutation)

        # scale
        if self.value_range_min != 0.0 or self.value_range_max != 1.0:
            x *= self.value_range_max - self.value_range_min  # [0, max_val - min_val]
            x += self.value_range_min  # [min_val, max_val]

        # crop -> resize -> augment
        if transform:
            x = self.transform(x)
        return x

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        r"""
        Converts a normalized tensor of an image to a denormalized numpy array.
        Output: np.uint8, shape: [..., h, w, c], range: [0, 255]

        Args:
            x (torch.Tensor): Input tensor of shape [..., c, h, w] and (approx.) range [min_val, max_val].

        Returns: A post-processed (quantized) sequence array ready for display.
        """

        # assuming shape = [..., c, h, w] -> [..., h, w, c]
        if x.ndim < 3:
            raise ValueError(f"expected at least three dimensions for input image")
        else:
            permutation = list(range(x.ndim - 3)) + [-2, -1, -3]
            x = x.permute(permutation)

        x -= self.value_range_min  # ~[0, max_val - min_val]
        x /= self.value_range_max - self.value_range_min  # ~[0, 1]
        x *= 255.  # ~[0, 255]
        x = torch.clamp(x, 0., 255.)
        x = x.cpu().numpy().astype('uint8')
        return x

    def default_available(self, split: str, **dataset_kwargs):
        r"""
        Tries to load a dataset and a datapoint using the default :attr:`self.data_dir` value.
        If this succeeds, then we can safely use the default data dir,
        otherwise a new dataset has to be downloaded and prepared.

        Args:
            split (str): The dataset's split identifier (i.e. whether it's a training/validation/test dataset).
            **dataset_kwargs (Any): Optional dataset arguments for image transformation, value_range, splitting etc.

        Returns: True if we could load the dataset using default values, False otherwise.

        """
        try:
            kwargs_ = deepcopy(dataset_kwargs)
            kwargs_.update({"data_dir": self.DEFAULT_DATA_DIR})
            default_ = self.__class__(split, **kwargs_)
            default_.set_seq_len(1, 1, 1)
            _ = default_[0]
        except (FileNotFoundError, ValueError, IndexError) as e:  # TODO other exceptions?
            return False
        return True

    @classmethod
    def download_and_prepare_dataset(cls):
        r"""
        Downloads the specific dataset, prepares it for the video prediction task (if needed)
        and stores it in a default location in the 'data/' folder.
        Implemented by the derived dataset classes.
        """
        raise NotImplementedError

    @classmethod
    def get_train_val(cls, **dataset_kwargs):
        r"""
        A wrapper method that creates a training and a validation dataset from the given dataset class.
        Like when initializing such datasets directly,
        optional dataset arguments can be specified with `\*\*dataset_kwargs`.

        Args:
            **dataset_kwargs (Any): Optional dataset arguments for image transformation, value_range, splitting etc.

        Returns: The created training and validation dataset of the same class.

        """
        assert cls.VALID_SPLITS == ["train", "test"] or cls.VALID_SPLITS == ["train", "val", "test"], \
            f"parameter 'VALID_SPLITS' of dataset class '{cls.__name__}' is ill-configured"

        # CAUTION: datasets that need set_seq_len to be ready can't be split using _random_split()
        if cls.VALID_SPLITS == ["train", "test"]:
            D_main = cls("train", **dataset_kwargs)
            len_main = len(D_main)
            len_train = int(len_main * cls.train_to_val_ratio)
            len_val = len_main - len_train
            D_train, D_val = _random_split(D_main, [len_train, len_val], cls.train_val_seed)
        else:
            D_train = cls("train", **dataset_kwargs)
            D_val = cls("val", **dataset_kwargs)
        return D_train, D_val

    @classmethod
    def get_test(cls, **dataset_kwargs):
        r"""
        A wrapper method that creates a test dataset from the given dataset class.
        Like when initializing such datasets directly,
        optional dataset arguments can be specified with `\*\*dataset_kwargs`.

        Args:
            **dataset_kwargs (Any): optional dataset arguments for image transformation, value_range, splitting etc.

        Returns: The created test dataset of the same class.

        """
        D_test = cls("test", **dataset_kwargs)
        return D_test


def _random_split(dataset: VPDataset, lengths: Sequence[int], random_seed: int) -> List[VPSubset]:
    r"""
    Custom implementation of torch.utils.data.random_split that returns SubsetWrappers.

    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results.

    Args:
        dataset (Dataset): Dataset to be split.
        lengths (sequence): lengths of splits to be produced.
        random_seed (int): RNG seed used for the random permutation.

    Returns:
        A list of VPSubsets containing the randomly split datasets.
    """

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    random.Random(random_seed).shuffle(indices)
    return [VPSubset(dataset, indices[offset - length: offset])
            for offset, length in zip(_accumulate(lengths), lengths)]
