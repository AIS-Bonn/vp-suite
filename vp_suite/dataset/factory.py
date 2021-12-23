import sys
sys.path.append("")

from torch.utils.data import DataLoader, Subset

from vp_suite.dataset.dataset_mmnist import MovingMNISTDataset
from vp_suite.dataset.dataset_bair import BAIRPushingDataset
from vp_suite.dataset.dataset_kth import KTHActionsDataset
from vp_suite.dataset.dataset_synpick import SynpickVideoDataset

# =====================================================================================================================

def create_train_val_dataset(cfg):
    dataset_class = dataset_classes.get(cfg.dataset, "MM")
    D_train, D_val = dataset_class.get_train_val(cfg)
    L_train = DataLoader(D_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    L_val = DataLoader(D_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    return (D_train, D_val), (L_train, L_val)

def create_test_dataset(cfg):
    dataset_class = dataset_classes.get(cfg.dataset, "MM")
    D_test = dataset_class.get_test(cfg)
    L_test = DataLoader(D_test, batch_size=1, shuffle=True, num_workers=0)
    return D_test, L_test

def update_cfg_from_dataset(cfg, D):
    D_ = D.dataset if type(D) == Subset else D
    cfg.action_size = D_.ACTION_SIZE
    if cfg.action_size < 1 and cfg.use_actions:
        print(f"INFO: dataset {D_.NAME} doesn't support actions -> action-conditioning is turned off.")
        cfg.use_actions = False
    cfg.img_h, cfg.img_w, cfg.img_c = D_.DEFAULT_FRAME_SHAPE
    cfg.img_shape = cfg.img_c, cfg.img_h, cfg.img_w
    return cfg

dataset_classes = {
    "MM": MovingMNISTDataset,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPV": SynpickVideoDataset
}

AVAILABLE_DATASETS = dataset_classes.keys()
