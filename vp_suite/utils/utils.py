import sys
from typing import List
from datetime import datetime
import subprocess
import shlex
import inspect
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np


def most(l: List[bool], factor=0.67):
    '''
    Like List.all(), but not 'all' of them.
    '''
    return sum(l) >= factor * len(l)

def timestamp(program):
    """ Obtains a timestamp of the current system time in a human-readable way """

    timestamp = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    return f"{program}_{timestamp}"

def run_command(command, print_to_console=True):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, encoding='utf8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output and print_to_console:
            print(output.strip())
    rc = process.poll()
    return rc

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

def _check_optuna_config(optuna_cfg : dict):

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

def set_from_kwarg(obj, attr_name, attr_default, kwarg_dict, required=False, choices=None):
    if required and attr_name not in kwarg_dict.keys():
        raise ValueError(f"missing required parameter '{attr_name}'")

    # check type fit
    attr_val = kwarg_dict.get(attr_name, attr_default)
    if not isinstance(attr_val, type(attr_default)):
        raise ValueError(f"mismatching types for parameter '{attr_name}'")

    # if choices are given, check if val matches choice
    if choices is not None:
        # If multiple args are given, check each one of them
        if isinstance(attr_val, list) and not isinstance(choices, list):
            for i, val_ in enumerate(attr_val):
                if val_ not in choices:
                    raise ValueError(f"entry {i} of parameter '{attr_name}' is not "
                                     f"one of the acceptable choices ({choices})")
        # else, check if single argument is valid choice
        elif attr_val not in choices:
            raise ValueError(f"parameter '{attr_name}' is not one of the acceptable choices ({choices})")
    setattr(obj, attr_name, attr_val)

def read_mp4(filepath: Path):
    fp = str(filepath.resolve())
    cap = cv2.VideoCapture(fp)
    if not cap.isOpened():
        raise ValueError(f"opening MP4 file '{fp}' failed")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return np.stack(frames, axis=0)   # [t, h, w, c]

def get_public_attrs(obj, calling_fn_name: str):
    r"""
    Similar to inspect.getmembers()

    Args:
        obj ():
        calling_fn_name ():

    Returns:

    """
    attr_dict = dict()
    instance_names = set(dir(obj))
    instance_names = [n for n in instance_names if not n.startswith("_")]  # remove private fields and dunders
    instance_names.remove(calling_fn_name)  # remove name of calling function to avoid recursion
    for name in instance_names:
        value = getattr(obj, name)
        if not inspect.isroutine(value):  # remove routines
            attr_dict[name] = value
    return attr_dict