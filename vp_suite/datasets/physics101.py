import os
import random
import torch
from torchvision.io import read_video
from pathlib import Path

from vp_suite.base import VPDataset, VPData
from vp_suite.defaults import SETTINGS
from vp_suite.utils.utils import set_from_kwarg, read_video


class Physics101Dataset(VPDataset):
    r"""
    Dataset class for the Videos of the dataset "Physics 101", as encountered in
    "Physics 101: Learning Physical Object Properties from Unlabeled Videos" by Wu et al.
    (http://phys101.csail.mit.edu/papers/phys101_bmvc.pdf).

    Each sequence depicts object-centered physical properties by showing objects of various materials and apperances in
    different physical scenarios such as sliding down a ramp or bouncing off a flat surface.
    """
    NAME = "Physics 101"
    REFERENCE = "http://phys101.csail.mit.edu/"
    IS_DOWNLOADABLE = "Yes"
    DEFAULT_DATA_DIR = SETTINGS.DATA_PATH / "phys101"
    AVAILABLE_CAMERAS = ["Camera_1", "Camera_2", "Kinect_RGB_1"]  #: Available cameras/image sources.
    AVAILABLE_SUBSEQ = ["start", "middle", "end"]  #: Available (sub-)sequence extraction position identifiers.
    MIN_SEQ_LEN = 16
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (1080, 1920, 3)

    camera = "Kinect_RGB_1"  #: Which camera to use from the dataset.
    subseq = "middle"  #: Where to extract the sequence from: "start" starts from the first frame, "end" ends at the last frame and "middle" lies exactly in between.
    trainval_to_test_ratio = 0.8  #: The ratio of files that will be training/validation data (rest will be test data).
    trainval_test_seed = 1612 #: The random seed used to separate training/validation and testing data. Value from the 'Noether Networks' code

    def __init__(self, split, **dataset_kwargs):
        super(Physics101Dataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["vid_filepaths"])

        # set attributes
        set_from_kwarg(self, dataset_kwargs, "camera", choices=self.AVAILABLE_CAMERAS)
        set_from_kwarg(self, dataset_kwargs, "subseq", choices=self.AVAILABLE_SUBSEQ)
        set_from_kwarg(self, dataset_kwargs, "trainval_test_seed")

        # get video filepaths for train/val or test
        self.vid_filepaths: [Path] = sorted(list(Path(self.data_dir).rglob(f"**/{self.camera}.mp4")))
        slice_idx = int(len(self.vid_filepaths) * self.trainval_to_test_ratio)
        random.Random(self.trainval_test_seed).shuffle(self.vid_filepaths)
        if self.split == "train":
            self.vid_filepaths = self.vid_filepaths[:slice_idx]
        else:
            self.vid_filepaths = self.vid_filepaths[slice_idx:]

    def __getitem__(self, i) -> VPData:
        # loaded video shape: [T, h, w, c], sitting in index 0 of the object returned by read_video()
        vid_fp = self.vid_filepaths[i]
        vid = read_video(vid_fp, num_frames=self.total_frames)  # [T, h, w, c]
        if self.seq_step > 1:
            vid = vid[::self.seq_step]  # [t, h, w, c]

        if self.subseq == "start":
            vid = vid[:self.total_frames]
        elif self.subseq == "end":
            vid = vid[-self.total_frames:]
        if self.subseq == 'middle':
            frame_offset = (vid.shape[0] - self.total_frames) // 2
            vid = vid[frame_offset : frame_offset+self.total_frames]

        vid = self.preprocess(vid)
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = {"frames": vid, "actions": actions, "origin": f"{vid_fp}, subseq mode: {self.subseq}"}
        return data

    def __len__(self):
        return len(self.vid_filepaths)

    def download_and_prepare_dataset(self):
        d_path = self.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)
        vid_filepaths: [Path] = list(d_path.rglob(f"**/*.mp4"))
        if len(vid_filepaths) == 0:  # no data available -> unpack tar
            tar_fname = "phys101_v1.0.tar"
            tar_path = str(d_path / tar_fname)
            if not os.path.exists(tar_path):  # no tar available -> download it
                URL = f"http://phys101.csail.mit.edu/data/{tar_fname}"
                from vp_suite.utils.utils import download_from_url
                download_from_url(URL, tar_path)

            print("Extracting data...")
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(d_path)
            tar.close()
            os.remove(tar_path)
