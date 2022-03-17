import sys
from typing import List, Union
from datetime import datetime
import subprocess
import inspect
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn


def most(l: List[bool], factor: float = 0.67):
    r"""
    Args:
        l (List[bool]): The input list to check.
        factor (float): The fraction that needs to be surpassed.

    Returns:
        Similar to List.all(), returns True if more than the specified fraction of entries is true in given list,
        False otherwise (e.g. for factor=0.5, returns True if at least half the entries of l are True.
    """
    return sum(l) >= factor * len(l)


def timestamp(program: str):
    """
    Args:
        program (str): A string identifier specifying which run is asking for the timestamp.

    Returns: A timestamp of the current system time in a human-readable way, prepended by the given program string.
    """
    timestamp = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    return f"{program}_{timestamp}"


def run_shell_command(command: str):
    r"""
    Runs the given command in the shell.

    Args:
        command (str): The given command as a single string.
    """
    subprocess.check_call(command, shell=True)


class TqdmUpTo(tqdm):
    r"""
    A wrapper class around the tqdm progress bar that can be used for showing download progress.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Taken from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5,
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [mentioned here](https://github.com/pypa/twine/commit/42e55e06).
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        r"""
        Updates the tqdm progress indicator.
        b (int): Number of blocks transferred so far [default: 1].
        bsize (int): Size of each block (in tqdm units) [default: 1].
        tsize (int): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_from_url(url: str, dst_path : str):
    r"""
    Downloads the contents of specified URL to the specified destination filepath.
    Uses :class:`TqdmUpTo` to show download progress.

    Args:
        url (str): The URL to download from.
        dst_path (str): The path to save the downloaded data to.
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    filename = url.split("/")[-1]
    print(f"Downloading from {url}...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urlretrieve(url, dst_path, reporthook=t.update_to)


def check_optuna_config(optuna_cfg: dict):
    r"""
    Checks whether the syntax of given optuna configuration dict is correct.

    Args:
        optuna_cfg (dict): The optuna configuration to be checked.
    """
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


def set_from_kwarg(obj, kwarg_dict: dict, attr_name: str,
                   default=None, required=False, choices=None, skip_unusable=False):
    r"""
    Sets an attribute in given object by using given name, configuration dict and utility values.

    Args:
        obj (Any): The object in which the attribute should be set.
        kwarg_dict (dict): The configuration dict where the attribute should be taken from.
        attr_name (str): The attribute name.
        default (Any): If the given config. dict does not contain the given attribute name as key, this default value will be used instead.
        required (bool): If set to True, will raise an Error rather than using the default value.
        choices (Any): If specified, checks whether the value(s) to be set is (are) one of the valid choices.
        skip_unusable (bool): If specified, skips attributes that are not already defined in the object.
    """
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
    r"""
    Reads and returns the video specified by given file path as a numpy array.

    Args:
        fp (Union[Path, str]): The filepath to read the video from.
        img_size ((int, int)): The desired frame size (height and width; frames will be reshaped to this size)
        start_index (int): Index of first frame to read.
        num_frames (int): Nmber of frames to read (default value -1 signifies that video is read to the end).

    Returns: The read video as a numpy array of shape (frames, height, width, channels).
    """
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
    r"""
    Args:
        fp (Union[Path, str]): The filepath of the video to be checked.

    Returns: The number of frames in the video at given filepath
    """
    if isinstance(fp, Path):
        fp = str(fp.resolve())
    cap = cv2.VideoCapture(fp)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def get_public_attrs(obj, calling_method: str = None, non_config_vars: List[str] = None, model_mode: bool = False):
    r"""
    Similarly to inspect.getmembers(), this method returns a dictionary containing all public attributes of an object.

    Args:
        obj (Any): The object to check.
        calling_method (str): The name of the calling method. If e.g. this gets called from one of the object's property fields, it has to be excluded to avoid recursion.
        non_config_vars(List[str]): A list of the attributes that should not be included in the result.
        model_mode(bool): If set to True, disregards any attributes that are torch.nn.Module or torch.Tensor objects.

    Returns: A dictionary containing all public attributes of an object.
    """
    attr_dict = dict()
    instance_names = set(dir(obj))
    instance_names = [n for n in instance_names if not n.startswith("_") and not n[0].isupper()]  # remove private fields, dunders and constants (starting with capital letter)
    if calling_method is not None:
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


class TimeOutException(Exception):
    r"""
    A custom Exception type thrown when signals time out (e.g. for timed input prompts).
    """
    pass


def alarm_handler(signum, frame):
    r"""
    A simple alarm handler that raises a :class:`TimeoutException`.
    """
    raise TimeOutException()


def timed_input(description: str, default=None, secs: int = 60):
    r"""
    A wrapper around the default `input()` statement, imposing a time limit and providing default values if
    the input is empty.
    Args:
        description (str): A description text that will be displayed with the input prompt.
        default (Any): The default value to assign to the variable if the the input is empty.
        secs (int): Time limit in seconds.

    Returns: The input value (or the default value if input is empty).
    """
    import signal
    signal.signal(signal.SIGALRM, alarm_handler)
    try:
        signal.alarm(secs)
        value = input(f"{description} [{default}]: ") or default
    except TimeOutException:
        print("Time limit reached, using default value...")
        value = default
    signal.alarm(0)
    return value


class PytestExpectedException(Exception):
    r"""
    A custom exception type that, when raised during pytest execution, causes pytest to skip the current test.
    """
    pass
