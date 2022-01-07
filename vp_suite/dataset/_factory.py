from torch.utils.data import Subset

from vp_suite.dataset.mmnist import MovingMNISTDataset
from vp_suite.dataset.bair import BAIRPushingDataset
from vp_suite.dataset.kth import KTHActionsDataset
from vp_suite.dataset.synpick import SynpickVideoDataset

# =====================================================================================================================

def update_cfg_from_dataset(config, D):
    D_ = D.dataset if type(D) == Subset else D
    config["action_size"] = D_.ACTION_SIZE
    if config["action_size"] < 1 and config["use_actions"]:
        print(f"INFO: dataset {D_.NAME} doesn't support actions -> action-conditioning is turned off.")
        config["use_actions"] = False
    config["img_h"], config["img_w"], config["img_c"] = D_.DEFAULT_FRAME_SHAPE
    config["img_shape"] = config["img_c"], config["img_h"], config["img_w"]
    return config

DATASET_CLASSES = {
    "MM": MovingMNISTDataset,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPV": SynpickVideoDataset
}

AVAILABLE_DATASETS = DATASET_CLASSES.keys()
