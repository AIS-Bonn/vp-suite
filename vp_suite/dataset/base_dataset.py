import torch
from torch.utils.data.dataset import Dataset
from typing import TypedDict

class VPData(TypedDict):
    frames: torch.Tensor  # shape: [t, c, h, w]
    actions: torch.Tensor  # shape: [t, a]

class BaseVPDataset(Dataset):

    def __init__(self, data_dir, cfg):

        super(BaseVPDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_cfg = cfg
        self.img_processor = cfg.img_processor
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i) -> VPData:
        raise NotImplementedError

    def preprocess_img(self, img):
        return self.img_processor.preprocess_img(img)