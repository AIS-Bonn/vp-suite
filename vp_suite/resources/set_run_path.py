r"""
An interactive python script that lets you change the save location of datasets,
models and logs to wherever you want them to be (also migrates existing data).
"""

from pathlib import Path
import json
import shutil

from vp_suite.utils.utils import timed_input
from vp_suite.defaults import SETTINGS

if __name__ == '__main__':
    local_conf_fp: str = SETTINGS.LOCAL_CONFIG_FP
    # obtain the current run path from the package installation config file
    try:
        with open(local_conf_fp, "r") as local_conf_file:
            old_run_path = Path(json.load(local_conf_file)["run_path"])
    except FileNotFoundError as e:
        print(f"file {local_conf_fp} does not exist yet -> creating...")
        old_run_path = SETTINGS.DEFAULT_RUN_PATH
    old_run_path_str = str(old_run_path.resolve())

    # get new run path from user input
    new_run_path = Path(timed_input("Please enter new run path", old_run_path_str))
    new_run_path_str = str(new_run_path.resolve())
    if old_run_path_str == new_run_path_str:
        print("Run path did not change. Exiting...")
        exit(0)

    if new_run_path.exists():
        raise ValueError("Provided Path exists already! Exiting...")

    # move existing run directory to new location
    if old_run_path.exists():
        print("moving existing data to new destination...")
        shutil.move(str(old_run_path.resolve()), new_run_path_str)

    # save new run path to local config file
    with open(local_conf_fp, "w") as local_conf_file:
        print(f"Writing new run path to {local_conf_fp}...")
        json.dump({"run_path": new_run_path_str}, local_conf_file)
