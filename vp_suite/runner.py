import sys, random, json
sys.path.append("")
import numpy as np
from copy import deepcopy

import torch

import vp_suite.constants as constants
from vp_suite.utils.img_processor import ImgProcessor
from vp_suite.dataset._factory import DATASET_CLASSES

class Runner:

    DEFAULT_RUN_CONFIG = (constants.PKG_RESOURCES / 'run_config.json').resolve()

    def __init__(self, device="cpu"):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.img_processor = ImgProcessor(value_min=0.0, value_max=1.0)

        self.reset_models()
        self.reset_datasets()

    @property
    def dataset_config(self):
        return None if self.dataset is None else self.dataset.get_config()

    def reset_datasets(self):
        self._reset_datasets()
        self.dataset = None
        self.datasets_ready = False

    def reset_models(self):
        self._reset_models()
        self.models_ready = False

    def load_dataset(self, dataset="MM", value_min=0.0, value_max=1.0, **dataset_kwargs):
        """
        ATTENTION: this removes any loaded models and datasets
        """
        self._reset_datasets()
        self._reset_models()
        self.img_processor.value_min = value_min
        self.img_processor.value_max = value_max
        dataset_class = DATASET_CLASSES[dataset]
        self._load_dataset(dataset_class, **dataset_kwargs)
        print(f"INFO: loaded dataset '{self.dataset.NAME}' from {self.dataset.data_dir} "
              f"(action size: {self.dataset.ACTION_SIZE})")
        self.datasets_ready = True

    def _reset_models(self):
        raise NotImplementedError

    def _reset_datasets(self):
        raise NotImplementedError

    def _load_dataset(self, dataset_class, **dataset_kwargs):
        raise NotImplementedError

    def _prepare_run(self, **run_args):

        with open(self.DEFAULT_RUN_CONFIG, 'r') as tc_file:
            run_config = json.load(tc_file)

        # update config
        assert all([run_arg in run_config.keys() for run_arg in run_args.keys()]), \
            f"Only the following run arguments are supported: {run_config.keys()}"
        run_config.update(run_args)

        # seed
        random.seed(run_config["seed"])
        np.random.seed(run_config["seed"])
        torch.manual_seed(run_config["seed"])

        return run_config
