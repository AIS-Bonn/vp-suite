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
    D = D_train.dataset if type(D_train) == Subset else D_train

    cfg.action_size = D.ACTION_SIZE
    cfg.img_h, cfg.img_w, cfg.img_c = D.DEFAULT_FRAME_SHAPE
    cfg.img_shape = cfg.img_c, cfg.img_h, cfg.img_w

    L_train = DataLoader(D_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    L_val = DataLoader(D_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    return (D_train, D_val), (L_train, L_val)

def create_test_dataset(cfg):
    dataset_class = dataset_classes.get(cfg.dataset, "MM")
    D_test = dataset_class.get_test(cfg)

    cfg.action_size = D_test.ACTION_SIZE
    cfg.img_h, cfg.img_w, cfg.img_c = D_test.DEFAULT_FRAME_SHAPE
    cfg.img_shape = cfg.img_c, cfg.img_h, cfg.img_w

    L_test = DataLoader(D_test, batch_size=1, shuffle=True, num_workers=0)
    return D_test, L_test


dataset_classes = {
    "MM": MovingMNISTDataset,
    "BAIR": BAIRPushingDataset,
    "KTH": KTHActionsDataset,
    "SPV": SynpickVideoDataset
}

AVAILABLE_DATASETS = dataset_classes.keys()
