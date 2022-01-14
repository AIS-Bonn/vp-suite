from pathlib import Path

import math
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from vp_suite.dataset._base_dataset import BaseVPDataset, VPData
import vp_suite.constants as constants

class MovingMNISTDataset(BaseVPDataset):

    NAME = "Moving MNIST"
    DEFAULT_DATA_DIR = constants.DATA_PATH / "moving_mnist"

    max_seq_len = 20  # default MMNIST sequences span 20 frames
    action_size = 0
    frame_shape = (64, 64, 3)
    train_keep_ratio = 0.96  # big dataset -> val can be smaller

    def __init__(self, split, img_processor, **dataset_kwargs):
        super(MovingMNISTDataset, self).__init__(split, img_processor, **dataset_kwargs)

        self.data_dir = str((Path(self.data_dir) / split).resolve())
        self.data_ids = sorted(os.listdir(self.data_dir))
        self.data_fps = [os.path.join(self.data_dir, image_id) for image_id in self.data_ids]

    def __len__(self):
        return len(self.data_fps)

    def __getitem__(self, i) -> VPData:

        assert self.ready_for_usage, \
            "Dataset is not yet ready for usage (maybe you forgot to call set_seq_len())."

        rgb_raw = np.load(self.data_fps[i])  # [t', h, w]
        rgb_raw = np.expand_dims(rgb_raw, axis=-1).repeat(3, axis=-1) # [t', h, w, c]
        rgb = self.preprocess_img(rgb_raw)
        rgb = rgb[:self.seq_len:self.seq_step]  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = { "frames": rgb, "actions": actions }
        return data

    def download_and_prepare_dataset(self):

        frame_size = (64, 64)
        num_frames = 20  # length of each sequence
        digit_size = 28  # size of mnist digit within frame
        digits_per_image = 2  # number of digits in each frame
        d_path = self.DEFAULT_DATA_DIR
        d_path.mkdir(parents=True)

        # training sequences
        train_seqs = 60000
        print("generating training set...")
        train_data = generate_moving_mnist(d_path, training=True, shape=frame_size, num_frames=num_frames,
                                           num_images=train_seqs, original_size=digit_size,
                                           nums_per_image=digits_per_image)
        print("saving training set...")
        save_generated_mmnist(train_data, train_seqs, frame_size, d_path / "train")

        # testing sequences
        test_seqs = 10000
        print("generating test set...")
        test_data = generate_moving_mnist(d_path, training=False, shape=frame_size, num_frames=num_frames,
                                          num_images=test_seqs, original_size=digit_size,
                                          nums_per_image=digits_per_image)
        print("saving test set...")
        save_generated_mmnist(test_data, test_seqs, frame_size, d_path / "test")


# === MMNIST data preparation tools ============================================

def save_generated_mmnist(data, seqs, frame_size, out_path):
    out_path.mkdir()
    num_frames = data.shape[0] // seqs
    data = data.reshape(seqs, num_frames, *frame_size)
    for i in tqdm(range(data.shape[0])):
        cur_out_fp = out_path / f"seq_{i:05d}.npy"
        np.save(str(cur_out_fp), data[i])

###########################################################################################
# BELOW: scripts to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee / Praateek Mahajan (port to python 3)
# saves in hdf5, npz, or jpg (individual frames) format
# (modified from https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe)
###########################################################################################

# helper functions
def arr_from_img(im, mean=0, std=1):
    '''

    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract

    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, index, mean=0, std=1):
    '''

    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


# loads mnist from web on demand
def load_dataset(d_path, training):
    from vp_suite.utils.utils import download_from_url
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            mnist_source = 'http://yann.lecun.com/exdb/mnist/'
            fname_ = filename.split("/")[-1]
            download_from_url(mnist_source + fname_, filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images(str(d_path / 'train-images-idx3-ubyte.gz'))
    return load_mnist_images(str(d_path / 't10k-images-idx3-ubyte.gz'))


def generate_moving_mnist(d_path, training=False, shape=(64, 64), num_frames=30, num_images=100, original_size=28,
                          nums_per_image=2):
    '''

    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_images: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.

    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height

    '''
    mnist = load_dataset(d_path, training)
    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_images, 1, width, height), dtype=np.uint8)

    for img_idx in tqdm(range(num_images)):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_image) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
        # Get a list containing two PIL images randomly sampled from the database
        mnist_images = [Image.fromarray(get_image_from_array(mnist, r, mean=0)).resize((original_size, original_size),
                                                                                       Image.ANTIALIAS) \
                        for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)])

        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
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
