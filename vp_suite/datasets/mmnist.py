from pathlib import Path

import math
import os
import re
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from vp_suite.utils.utils import timed_input
from vp_suite.base import VPDataset, VPData
from vp_suite.defaults import SETTINGS

class MovingMNISTDataset(VPDataset):
    r"""
    Dataset class for the dataset "Moving MNIST", as firstly encountered in
    "Unsupervised Learning of Video Representations using LSTMs" by Srivastava et al.
    (https://arxiv.org/pdf/1502.04681v3.pdf).

    Each sequence depicts two digits from the MNIST dataset moving linearly in front of a black background,
    occasionally bouncing off the wall and overlapping each other.

    For downloading and preparing the dataset, scripts have been developed by Tencia Lee,
    ported to python 3 by Praateek Mahajan (https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe)
    and further modified here.
    """

    NAME = "Moving MNIST"
    REFERENCE = "https://arxiv.org/abs/1502.04681v3"
    IS_DOWNLOADABLE = "Yes"
    DEFAULT_DATA_DIR = SETTINGS.DATA_PATH / "moving_mnist"
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (64, 64, 3)

    train_to_val_ratio = 0.96

    def __init__(self, split, **dataset_kwargs):
        super(MovingMNISTDataset, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["data_ids", "data_fps"])

        self.data_dir = str((Path(self.data_dir) / split).resolve())
        self.data_ids = sorted([fn for fn in os.listdir(self.data_dir) if re.match(r"seq_[0-9]+\.npy", fn)])
        self.data_fps = [os.path.join(self.data_dir, image_id) for image_id in self.data_ids]
        self.MIN_SEQ_LEN = np.load(self.data_fps[0]).shape[0]  # sequence length depends on generated dataset

    def __len__(self):
        return len(self.data_fps)

    def __getitem__(self, i) -> VPData:
        if not self.ready_for_usage:
            raise RuntimeError("Dataset is not yet ready for usage (maybe you forgot to call set_seq_len()).")

        data_fp = self.data_fps[i]
        rgb_raw = np.load(data_fp)  # [t', h, w]
        rgb_raw = np.expand_dims(rgb_raw, axis=-1).repeat(3, axis=-1)  # [t', h, w, c]
        rgb_raw = rgb_raw[:self.seq_len:self.seq_step]  # [t, h, w, c]
        rgb = self.preprocess(rgb_raw)

        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = {"frames": rgb, "actions": actions, "origin": data_fp}
        return data

    def download_and_prepare_dataset(self):

        # defaults
        frame_size = (64, 64)
        num_frames = 20  # length of each sequence
        digit_size = 28  # size of mnist digit within frame
        digits_per_image = 2  # number of digits in each frame
        train_seqs = 60000
        test_seqs = 10000

        num_frames = int(timed_input("Number of frames per sequence", default=num_frames))
        digit_size = int(timed_input("Pixel size of digit in frame", default=digit_size))
        digits_per_image = int(timed_input("Digits per image", default=digits_per_image))
        train_seqs = int(timed_input("Number of training sequences", default=train_seqs))
        test_seqs = int(timed_input("Number of test sequences", default=test_seqs))

        d_path = self.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True, exist_ok=True)

        # training sequences
        print("generating training set...")
        train_data = generate_moving_mnist(d_path, training=True, shape=frame_size, num_frames=num_frames,
                                           num_images=train_seqs, digit_size=digit_size,
                                           digits_per_image=digits_per_image)
        print("saving training set...")
        save_generated_mmnist(train_data, train_seqs, frame_size, d_path / "train")

        # testing sequences
        print("generating test set...")
        test_data = generate_moving_mnist(d_path, training=False, shape=frame_size, num_frames=num_frames,
                                          num_images=test_seqs, digit_size=digit_size,
                                          digits_per_image=digits_per_image)
        print("saving test set...")
        save_generated_mmnist(test_data, test_seqs, frame_size, d_path / "test")


# === MMNIST data preparation tools ============================================

def save_generated_mmnist(data: np.ndarray, seqs: int, frame_size: (int, int), out_path: Path):
    r"""
    Save generated data per-sequence to specified out path.

    Args:
        data (np.ndarray): The generated data to save.
        seqs (int): The number of generated sequences.
        frame_size ((int, int)): The frame size.
        out_path (Path): The path where the data should be saved.
    """
    out_path.mkdir()
    num_frames = data.shape[0] // seqs
    data = data.reshape((seqs, num_frames, *frame_size))
    for i in tqdm(range(data.shape[0])):
        cur_out_fp = out_path / f"seq_{i:05d}.npy"
        np.save(str(cur_out_fp), data[i])


# helper functions
def arr_from_img(im, mean: float = 0, std: float = 1):
    r"""
    Convert image to array.

    Args:
        im(): Image.
        mean(float): Mean to subtract.
        std(float): Standard Deviation to subtract.

    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.

    """
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def img_from_arr(arr: np.ndarray, index: int, mean: float = 0, std: float = 1):
    r"""
    Convert array to image.

    Args:
        arr(np.ndarray): Dataset of shape N x C x W x H.
        index(int): Index of image we want to fetch.
        mean(float): Mean to add.
        std(float): Standard Deviation to add.

    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image.
    """
    ch, w, h = arr.shape[1], arr.shape[2], arr.shape[3]
    ret = (((arr[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


def load_dataset(d_path: Path, training: bool, digit_size: int):
    r"""
    Loads MNIST from the web on demand.

    Args:
        d_path (Path): The path where the downloaded digits should be stored.
        training (bool): Whether to use the training images (True) or the test images (False).
        digit_size (int): Size of the digit in pixels (height and width are the same)

    Returns: The loaded MNIST images.

    """
    from vp_suite.utils.utils import download_from_url
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            mnist_source = 'http://yann.lecun.com/exdb/mnist/'
            fname_ = filename.split("/")[-1]
            download_from_url(mnist_source + fname_, filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, digit_size, digit_size).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images(str(d_path / 'train-images-idx3-ubyte.gz'))
    return load_mnist_images(str(d_path / 't10k-images-idx3-ubyte.gz'))


def generate_moving_mnist(d_path: Path, training: bool, shape: (int, int), num_frames: int, num_images: int,
                          digit_size: int, digits_per_image: int):
    r"""
    Generate sequences of moving MNIST digits by moving them around between frames.

    Args:
        training (bool): Used to decide if downloading/generating training set or test set.
        shape ((int, int)): Shape we want for our moving images (new_width and new_height).
        num_frames (int): Number of frames in a particular movement/animation/gif.
        num_images (int): Number of movement/animations/gif to generate.
        digit_size (int): Real size of the images (eg: MNIST is 28x28).
        digits_per_image (int): Digits per movement/animation/gif.

    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height

    """
    mnist = load_dataset(d_path, training, digit_size)
    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - digit_size, height - digit_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_images, 1, width, height), dtype=np.uint8)

    for img_idx in tqdm(range(num_images)):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(digits_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=digits_per_image) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
        # Get a list containing two PIL images randomly sampled from the database
        mnist_images = [Image.fromarray(img_from_arr(mnist, r, mean=0)).resize((digit_size, digit_size),
                                                                               Image.ANTIALIAS) \
                        for r in np.random.randint(0, mnist.shape[0], digits_per_image)]
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(digits_per_image)])

        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(digits_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames + frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)
    return dataset
