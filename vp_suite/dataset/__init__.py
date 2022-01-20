r"""
This package contains the datasets.
"""

from vp_suite.dataset.bair import BAIRPushingDataset
from vp_suite.dataset.kth import KTHActionsDataset
from vp_suite.dataset.mmnist import MovingMNISTDataset
from vp_suite.dataset.synpick import SynpickVideoDataset

DATASET_CLASSES = {
    "MM": MovingMNISTDataset,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPV": SynpickVideoDataset
}  #: a mapping of all the datasets available for use.
AVAILABLE_DATASETS = DATASET_CLASSES.keys()
