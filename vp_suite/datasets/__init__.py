r"""
This package contains the datasets.
"""

from vp_suite.datasets.bair import BAIRPushingDataset
from vp_suite.datasets.kth import KTHActionsDataset
from vp_suite.datasets.mmnist import MovingMNISTDataset
from vp_suite.datasets.synpick import SynpickVideoDataset
from vp_suite.datasets.physics101 import Physics101Dataset

DATASET_CLASSES = {
    "MM": MovingMNISTDataset,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPV": SynpickVideoDataset,
    "P101": Physics101Dataset,
}  #: a mapping of all the datasets available for use.
AVAILABLE_DATASETS = DATASET_CLASSES.keys()
