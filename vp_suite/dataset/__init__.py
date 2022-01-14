from bair import BAIRPushingDataset
from kth import KTHActionsDataset
from mmnist import MovingMNISTDataset
from synpick import SynpickVideoDataset

DATASET_CLASSES = {
    "MM": MovingMNISTDataset,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPV": SynpickVideoDataset
}
AVAILABLE_DATASETS = DATASET_CLASSES.keys()
