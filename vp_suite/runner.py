import sys, random, json
sys.path.append("")
import numpy as np

import torch

import vp_suite.constants as constants
from vp_suite.utils.img_processor import ImgProcessor

class Runner:

    DEFAULT_RUN_CONFIG = (constants.PKG_RESOURCES / 'run_config.json').resolve()

    def __init__(self, device="cpu"):
        with open(self.DEFAULT_RUN_CONFIG, 'r') as tc_file:
            self.config = json.load(tc_file)
        self.config["device"] = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])

        self.img_processor = ImgProcessor(value_min=0.0, value_max=1.0)

        self._reset_models()
        self._reset_datasets()

    def load_dataset(self, dataset="MM", value_min=0.0, value_max=1.0, **dataset_kwargs):
        raise NotImplementedError

    def _reset_models(self):
        raise NotImplementedError

    def _reset_datasets(self):
        raise NotImplementedError
