import sys
from typing import List, Union
from datetime import datetime
import subprocess
import shlex
import inspect
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn


def most(l: List[bool], factor=0.67):
    '''
    Like List.all(), but not 'all' of them.
    '''
    return sum(l) >= factor * len(l)


def timestamp(program):
    """ Obtains a timestamp of the current system time in a human-readable way """

    timestamp = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    return f"{program}_{timestamp}"


def run_shell_command(command):
    subprocess.check_call(command, shell=True)


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Taken from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5,
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_from_url(url: str, dst_path : str):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    filename = url.split("/")[-1]
    print(f"Downloading from {url}...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urlretrieve(url, dst_path, reporthook=t.update_to)


def check_optuna_config(optuna_cfg : dict):

    try:
        for parameter, p_dict in optuna_cfg.items():
            if not isinstance(p_dict, dict):
                raise ValueError
            if "choices" in p_dict.keys():
                if not isinstance(p_dict["choices"], list):
                    raise ValueError
            else:
                if not {"type", "min", "max"}.issubset(set(p_dict.keys())):
                    raise ValueError
                if p_dict["min"] > p_dict["max"]:
                    raise ValueError
                if p_dict["type"] == "float" and p_dict.get("scale", '') not in ["log", "uniform"]:
                    raise ValueError
    except ValueError:
        print("invalid optuna config")


def set_from_kwarg(obj, kwarg_dict, attr_name, default=None, required=False, choices=None, skip_unusable=False):

    # required parameter?
    if required and attr_name not in kwarg_dict.keys():
        raise ValueError(f"missing required parameter '{attr_name}' for object '{obj.__class__}'")

    # skip if not existant in obj?
    if skip_unusable and not hasattr(obj, attr_name):
        print(f"parameter '{attr_name}' is not usable for init of object '{obj.__class__}' -> skipping")

    # get default if available
    if default is None and hasattr(obj, attr_name):
        default = getattr(obj, attr_name)

    # check type fit
    attr_val = kwarg_dict.get(attr_name, default)
    if default is not None and not isinstance(attr_val, type(default)):
        raise ValueError(f"mismatching types for parameter '{attr_name}' for object '{obj.__class__}'")

    # if choices are given, check if val matches choice
    if choices is not None:
        # If multiple args are given, check each one of them
        if isinstance(attr_val, list):
            for i, val_ in enumerate(attr_val):
                if val_ not in choices:
                    raise ValueError(f"entry {i} of parameter '{attr_name}' is not "
                                     f"one of the acceptable choices ({choices})")
        # else, check if single argument is valid choice
        elif attr_val not in choices:
            raise ValueError(f"parameter '{attr_name}' is not one of the acceptable choices ({choices})")

    setattr(obj, attr_name, attr_val)


def read_video(fp: Union[Path, str], img_size: (int, int) = None,
               start_index=0, num_frames=-1):

    if isinstance(fp, Path):
        fp = str(fp.resolve())
    cap = cv2.VideoCapture(fp)
    if not cap.isOpened():
        raise ValueError(f"opening MP4 file '{fp}' failed")

    collected_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    while num_frames < 0 or len(collected_frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        collected_frames.append(frame)
    cap.release()

    if img_size is not None:
        h, w = img_size
        collected_frames = [cv2.resize(frame, (w, h)) for frame in collected_frames]
    collected_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in collected_frames]
    return np.stack(collected_frames, axis=0)   # [t, h, w, c]


def get_frame_count(fp: Union[Path, str]):
    if isinstance(fp, Path):
        fp = str(fp.resolve())
    cap = cv2.VideoCapture(fp)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def get_public_attrs(obj, calling_method: str, non_config_vars: [str] = None, model_mode: bool = False):
    r"""
    Similar to inspect.getmembers()

    Args:
        obj ():
        calling_method (str):
        non_config_vars([str]):
        model_mode(bool):

    Returns:

    """
    attr_dict = dict()
    instance_names = set(dir(obj))
    instance_names = [n for n in instance_names if not n.startswith("_")]  # remove private fields and dunders
    instance_names.remove(calling_method)  # remove name of calling method to avoid recursion
    for name in instance_names:
        value = getattr(obj, name)
        if inspect.isroutine(value):  # disregard routines
            continue
        if model_mode and (isinstance(value, nn.Module) or isinstance(value, torch.Tensor)):  # disregard nn.Module objects and tensors if specified
            continue
        attr_dict[name] = value
    for k in (non_config_vars or []):  # remove non-config vars, if any
        attr_dict.pop(k, None)
    return attr_dict
