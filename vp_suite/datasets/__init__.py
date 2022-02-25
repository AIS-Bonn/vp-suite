r"""
This package contains the all datasets available for use. See docs section "Available Datasets" for an overview.
"""

from vp_suite.datasets.bair import BAIRPushingDataset
from vp_suite.datasets.kth import KTHActionsDataset
from vp_suite.datasets.mmnist import MovingMNISTDataset
from vp_suite.datasets.mmnist_on_the_fly import MovingMNISTOnTheFly
from vp_suite.datasets.synpick import SynpickMovingDataset
from vp_suite.datasets.physics101 import Physics101Dataset
from vp_suite.datasets.human36m import Human36MDataset
from vp_suite.datasets.kitti_raw import KITTIRawDataset
from vp_suite.datasets.caltech_pedestrian import CaltechPedestrianDataset

DATASET_CLASSES = {
    "MM": MovingMNISTDataset,
    "MMF": MovingMNISTOnTheFly,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPM": SynpickMovingDataset,
    "P101": Physics101Dataset,
    "H36M": Human36MDataset,
    "KITTI": KITTIRawDataset,
    "CP": CaltechPedestrianDataset,
}
AVAILABLE_DATASETS = DATASET_CLASSES.keys()
