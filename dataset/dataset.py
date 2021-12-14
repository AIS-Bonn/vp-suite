import sys, os
sys.path.append(".")

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset_mmnist import MovingMNISTDataset
from dataset.dataset_bair import BAIRPushingDataset
from dataset.dataset_kth import KTHActionsDataset
from dataset.dataset_kth2 import KTHActionsDataset2
from dataset.synpick_vid import SynpickVideoDataset

def create_dataset_mm(cfg):
    data_dir = cfg.data_dir
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")
    D_main = MovingMNISTDataset(train_path)
    D_test = MovingMNISTDataset(test_path)
    len_train = int(len(D_main) * 0.96)
    len_val = len(D_main) - len_train
    D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
    return D_train, D_val, D_test

def create_dataset_bair(cfg):
    data_dir = os.path.join(cfg.data_dir, "softmotion30_44k")
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    D_main = BAIRPushingDataset(train_dir)
    D_test = BAIRPushingDataset(test_dir)
    len_train = int(len(D_main) * 0.96)
    len_val = len(D_main) - len_train
    D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
    return D_train, D_val, D_test

def create_dataset_kth(cfg):
    D_main = KTHActionsDataset(cfg.data_dir, person_ids=range(1, 17))
    D_test = KTHActionsDataset(cfg.data_dir, person_ids=range(17, 26))
    len_train = int(len(D_main) * 0.8)
    len_val = len(D_main) - len_train
    D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
    return D_train, D_val, D_test

def create_dataset_kth2(cfg):
    D_main = KTHActionsDataset2(cfg.data_dir, "train")
    D_test = KTHActionsDataset2(cfg.data_dir, "test")
    len_train = int(len(D_main) * 0.8)
    len_val = len(D_main) - len_train
    D_train, D_val = torch.utils.data.random_split(D_main, [len_train, len_val])
    return D_train, D_val, D_test

def create_dataset_synpick_vid(cfg):
    data_dir = cfg.data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_data = SynpickVideoDataset(data_dir=train_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap, num_classes=cfg.num_classes,
                                     include_gripper=cfg.include_gripper)
    val_data = SynpickVideoDataset(data_dir=val_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                   allow_overlap=cfg.vid_allow_overlap, num_classes=cfg.num_classes,
                                   include_gripper=cfg.include_gripper)
    test_data = SynpickVideoDataset(data_dir=test_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                    allow_overlap=cfg.vid_allow_overlap, num_classes=cfg.num_classes,
                                    include_gripper=cfg.include_gripper)
    return train_data, val_data, test_data

# =====================================================================================================================

def create_dataset(cfg):

    dataset_creator = dataset_creators.get(cfg.dataset, "MM")
    D_train, D_val, D_test = dataset_creator(cfg)
    D = D_train.dataset if type(D_train) == Subset else D_train
    cfg.action_size = D.action_size
    cfg.img_shape = D.img_shape
    #print(len(D))
    #exit(0)

    # loaders
    train_loader = DataLoader(D_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(D_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    test_loader = DataLoader(D_test, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    return (D_train, D_val, D_test), (train_loader, val_loader, test_loader)

dataset_creators = {
    "MM": create_dataset_mm,
    "BAIR": create_dataset_bair,
    "KTH": create_dataset_kth,
    "KTH2": create_dataset_kth2,
    "SPV": create_dataset_synpick_vid
}

SUPPORTED_DATASETS = dataset_creators.keys()
