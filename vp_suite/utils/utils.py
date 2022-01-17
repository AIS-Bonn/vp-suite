import sys
from typing import List
from datetime import datetime
import subprocess
import shlex

from tqdm import tqdm


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
            assert isinstance(p_dict, dict)
            if "choices" in p_dict.keys():
                assert (isinstance(p_dict["choices"], list))
            else:
                assert {"type", "min", "max"}.issubset(set(p_dict.keys()))
                assert p_dict["min"] <= p_dict["max"]
                if p_dict["type"] == "float":
                    assert p_dict.get("scale", '') in ["log", "uniform"]
    except AssertionError:
        print("ERROR: invalid optuna config")