import numpy as np
import torch
from torchvision.datasets import MNIST

from vp_suite.base import VPDataset, VPData
from vp_suite.defaults import SETTINGS


class MovingMNISTOnTheFly(VPDataset):
    r"""
    Dataset class for the dataset "Moving MNIST", as firstly encountered in
    "Unsupervised Learning of Video Representations using LSTMs" by Srivastava et al.
    (https://arxiv.org/pdf/1502.04681v3.pdf).

    Each sequence depicts two digits from the MNIST dataset moving linearly in front of a black background,
    occasionally bouncing off the wall and overlapping each other.

    As opposed to the other Moving MNIST dataset, this one generates the digit sequences on-the-fly,
    randomly sampling digits and velocities. Besides the digit templates, no data is downloaded.
    """
    NAME = "Moving MNIST - On the fly"
    IS_DOWNLOADABLE = "Yes (MNIST digits)"
    ON_THE_FLY = True
    DEFAULT_DATA_DIR = SETTINGS.DATA_PATH / "moving_mnist_on_the_fly"
    VALID_SPLITS = ["train", "val", "test"]
    MIN_SEQ_LEN = 1e8  #: Sequence length unbounded, depends on input sequence length
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (64, 64, 3)
    DEFAULT_N_SEQS = {"train": 9600, "val": 400, "test": 1000}  #: Default values for the dataset split sizes.
    SPLIT_SEED_OFFSETS = {"train": lambda x: 3*x+2, "val": lambda x: 3*x+1, "test": lambda x: 3*x}  #: passing the seed value to these functions guarantees unique RNG for all splits

    min_speed = 2
    max_speed = 5
    min_acc = 0
    max_acc = 0
    num_channels = 3
    num_digits = 2
    rng_seed = 4115  # with this default value, the test default becomes 3*x=12345.
    n_seqs = None

    def __init__(self, split, **dataset_kwargs):
        super(MovingMNISTOnTheFly, self).__init__(split, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["data", "rng", "digit_id_rng", "speed_rng", "acc_rng", "pos_rng",
                                     "get_digit_id", "get_speed", "get_acc", "get_init_pos"])

        if self.num_channels not in [1, 3]:
            raise ValueError("num_channels for dataset needs to be in [1, 3].")
        img_c, img_h, img_w = self.img_shape
        if img_h != img_w:
            raise ValueError("MMNIST only permits square images")
        self.DATASET_FRAME_SHAPE = (img_h, img_w, img_c)  # TODO dirty hack

        # loading data and rng
        self.data = MNIST(root=self.data_dir, train=(self.split == "train"), download=False)
        self.n_seqs = self.n_seqs or self.DEFAULT_N_SEQS[self.split]
        self.digit_id_rng, self.speed_rng, self.acc_rng, self.pos_rng = None, None, None, None
        self.reset_rng()

        self.get_digit_id = lambda: self.digit_id_rng.integers(len(self.data))
        self.get_speed = lambda: self.speed_rng.integers(-1*self.max_speed, self.max_speed+1)
        self.get_acc = lambda: self.acc_rng.integers(-1*self.max_acc, self.max_acc+1)
        self.get_init_pos = lambda digit_size: (self.pos_rng.integers(0, self.img_shape[1]-digit_size),
                                                self.pos_rng.integers(0, self.img_shape[2]-digit_size))

    def __len__(self):
        return self.n_seqs

    def reset_rng(self):
        r"""
        Creates RNG-based generation helpers for the on-the-fly generation, re-setting the RNG.
        """
        split_rng_seed = self.SPLIT_SEED_OFFSETS[self.split](self.rng_seed)
        self.digit_id_rng = np.random.default_rng(split_rng_seed)
        self.speed_rng = np.random.default_rng(split_rng_seed)
        self.acc_rng = np.random.default_rng(split_rng_seed)
        self.pos_rng = np.random.default_rng(split_rng_seed)

    def __getitem__(self, i) -> VPData:
        if not self.ready_for_usage:
            raise RuntimeError("Dataset is not yet ready for usage (maybe you forgot to call set_seq_len()).")

        digits, next_poses, speeds, digit_size = [], [], [], None
        for i in range(self.num_digits):
            digit, pos, speed, digit_size = self._sample_digit()
            digits.append(digit)
            next_poses.append(pos)
            speeds.append(speed)

        # generating sequence by moving the digit given velocity
        frames = np.zeros((self.seq_len, *self.DATASET_FRAME_SHAPE))
        for i, frame in enumerate(frames):
            for j, (digit, cur_pos, speed) in enumerate(zip(digits, next_poses, speeds)):
                speed, cur_pos = self._move_digit(speed=speed, cur_pos=cur_pos,
                                                  img_size=self.img_shape[1], digit_size=digit_size)
                speeds[j] = speed
                next_poses[j] = cur_pos
                cur_h, cur_w = cur_pos
                frame[cur_h:cur_h+digit_size, cur_w:cur_w+digit_size] += digit
            frames[i] = np.clip(frame, 0, 1)
        frames = self.preprocess(frames * 255)

        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic
        data = {"frames": frames, "actions": actions, "origin": "generated on-the-fly"}
        return data

    def _sample_digit(self):
        """
        Samples digit, initial position and speed.
        """
        digit_id = self.get_digit_id()
        cur_digit = np.array(self.data[digit_id][0]) / 255  # sample IDX, digit
        digit_size = cur_digit.shape[-1]
        cur_digit = cur_digit[..., np.newaxis]
        if self.num_channels == 3:
            cur_digit = np.repeat(cur_digit, 3, axis=-1)

        # obtaining position in original frame
        x_coord, y_coord = self.get_init_pos(digit_size)
        cur_pos = np.array([y_coord, x_coord])

        # generating sequence
        speed_x, speed_y, acc = None, None, None
        while speed_x is None or np.abs(speed_x) < self.min_speed:
            speed_x = self.get_speed()
        while speed_y is None or np.abs(speed_y) < self.min_speed:
            speed_y = self.get_speed()
        while acc is None or np.abs(acc) < self.min_acc:
            acc = self.get_acc()
        speed = np.array([speed_y, speed_x])

        return cur_digit, cur_pos, speed, digit_size

    def _move_digit(self, speed, cur_pos, img_size, digit_size):
        """
        Performs digit movement. Also produces bounces and makes appropriate changes.
        """
        next_pos = cur_pos + speed
        for i, p in enumerate(next_pos):
            # left/down bounce
            if p + digit_size > img_size:
                offset = p + digit_size - img_size
                next_pos[i] = p - offset
                speed[i] = -1 * speed[i]
            elif (p < 0):
                next_pos[i] = -1 * p
                speed[i] = -1 * speed[i]
        return speed, next_pos

    def download_and_prepare_dataset(self):
        r"""
        Downloads the MNIST digit data so that on-the-fly generation can take place.
        """
        _ = MNIST(root=self.DEFAULT_DATA_DIR, train=True, download=True)
        _ = MNIST(root=self.DEFAULT_DATA_DIR, train=False, download=True)
