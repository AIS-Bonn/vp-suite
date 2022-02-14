import math
from pathlib import Path

from torch import linalg as linalg

import torch
import torchvision.transforms.functional as TF

from vp_suite.measure.fvd._pytorch_i3d.pytorch_i3d import InceptionI3d
from vp_suite.base.base_measure import BaseMeasure

class FrechetVideoDistance(BaseMeasure):
    r"""
    INSPIRED BY: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

    """
    NAME = "Fréchet Video Distance (FVD)"
    REFERENCE = "https://arxiv.org/abs/1812.01717"

    _min_T = 9  #: TODO
    _max_T = 16  #: TODO
    _i3d_in_shape = (224, 224)  #: TODO
    _i3d_num_classes = 400  #: TODO
    _i3d_ckpt_file = "_pytorch_i3d/models/rgb_imagenet.pt"  #: TODO

    def __init__(self, device, in_channels=3):
        r"""

        Args:
            device ():
            in_channels ():
        """
        super(FrechetVideoDistance, self).__init__(device)

        self.i3d = InceptionI3d(num_classes=self._i3d_num_classes, in_channels=in_channels)
        self.i3d.load_state_dict(torch.load(Path(__file__).parent / self._i3d_ckpt_file))
        self.i3d.to(self.device)
        self.i3d.eval()  # don't train the pre-trained I3D
        self.to(self.device)

    @classmethod
    def valid_T(cls, x):
        r"""

        Args:
            x ():

        Returns:

        """
        return x >= cls._min_T and x <= cls._max_T

    def calculate_n_chunks(self, num_frames):
        r"""
        If given input length is too large, this function returns the number of chunks.
        Each chunk is then used for a separate fvd calculation, and their results are combined afterwards.

        Args:
            num_frames ():

        Returns:

        """
        n_chunks, drop_last_chunk = 1, False

        if num_frames < self._min_T:
            print(f"The I3D Module used for FVD needs at least"
                  f" {self._min_T} input frames (given: {num_frames}) -> returning None as loss value!")
            n_chunks = -1

        elif num_frames > self._max_T:
            possible_chunk_l = range(self._max_T, self._min_T - 1, -1)
            n_chunks = None
            for chunk_l in possible_chunk_l:
                if num_frames % chunk_l >= self._min_T:
                    n_chunks = num_frames // chunk_l + 1

            # loss-less chunking not possible -> get largest possible even chunk and drop last chunk
            if n_chunks is None:
                missed_frames = [num_frames % chunk_l for chunk_l in possible_chunk_l]
                best_chunk_l = sorted(zip(possible_chunk_l, missed_frames), key=lambda x: x[1])[-1]
                n_chunks = num_frames // best_chunk_l + 1
                drop_last_chunk = True

            print(f"The I3D Module used for FVD handles at most 16 input frames (given: {num_frames})"
                  f" -> input video will be consumed {n_chunks} chunks!")

        return n_chunks, drop_last_chunk

    def forward(self, pred, real):
        r"""
        input: [b, T, c, h, w]
        output: scalar

        Args:
            pred ():
            real ():

        Returns:

        """
        vid_shape = pred.shape
        if vid_shape != real.shape:
            raise ValueError("FrechetVideoDistance.get_distance(pred, real): vid shapes not equal!")

        num_frames = vid_shape[1]
        n_chunks, drop_last_chunk = self.calculate_n_chunks(num_frames)
        if n_chunks < 1:
            return None

        # resize images in video to 224x224 because the I3D network needs that
        pred = TF.resize(pred.reshape(-1, *vid_shape[2:]), self._i3d_in_shape)
        real = TF.resize(real.reshape(-1, *vid_shape[2:]), self._i3d_in_shape)

        # re-arrange dims for I3D input
        pred = pred.reshape(*vid_shape[:3], *self._i3d_in_shape).permute((0, 2, 1, 3, 4))  # [b, c, T, 224, 224]
        real = real.reshape(*vid_shape[:3], *self._i3d_in_shape).permute((0, 2, 1, 3, 4))

        pred_chunked = torch.chunk(pred, n_chunks, dim=2)
        real_chunked = torch.chunk(real, n_chunks, dim=2)

        n_valid_chunks = (n_chunks - 1) if drop_last_chunk else n_chunks
        chunk_distances = [self.get_distance(pred_chunked[i], real_chunked[i]) for i in range(n_valid_chunks)]
        return sum(chunk_distances) / n_valid_chunks  # mean

    def get_distance(self, pred, real):
        r"""
        input: [b, c, t, 224, 224] with t in suitable (chunked) size
        output: scalar

        Args:
            pred ():
            real ():

        Returns:

        """
        logits_pred = self.i3d.extract_features(pred).squeeze()  # [b, n]
        logits_real = self.i3d.extract_features(real).squeeze()

        if pred.shape[0] == 1:  # if batch size is 1, the prev. squeeze also removed the batch dim
            logits_pred = logits_pred.unsqueeze(dim=0)
            logits_real = logits_real.unsqueeze(dim=0)

        return calculate_2_wasserstein_dist(logits_pred, logits_real)


def calculate_2_wasserstein_dist(pred, real):
    r"""
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_real, P_pred = min_{X, Y} E[|X-Y|^2]

    For multivariate gaussian distributed inputs x_real ~ MN(mu_real, cov_real) and x_pred ~ MN(mu_pred, cov_pred),
    this reduces to: d = |mu_real - mu_pred|^2 - Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))

    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf

    Input shape: [b = batch_size, n = num_features]
    Output shape: scalar

    Args:
        pred ():
        real ():

    Returns:

    """
    if pred.shape != real.shape:
        raise ValueError("Expecting equal shapes for pred and real!")

    # the following ops need some extra precision
    pred, real = pred.transpose(0, 1).double(), real.transpose(0, 1).double()  # [n, b]
    mu_pred, mu_real = torch.mean(pred, dim=1, keepdim=True), torch.mean(real, dim=1, keepdim=True)  # [n, 1]
    n, b = pred.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_pred = pred - mu_pred
    E_real = real - mu_real
    cov_pred = torch.matmul(E_pred, E_pred.t()) * fact  # [n, n]
    cov_real = torch.matmul(E_real, E_real.t()) * fact

    # calculate Tr((cov_real * cov_pred)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues of the mm(cov_pred, cov_real) and therefore for M are real-valued.
    C_pred = E_pred * math.sqrt(fact)  # [n, n], "root" of covariance
    C_real = E_real * math.sqrt(fact)
    M_l = torch.matmul(C_pred.t(), C_real)
    M_r = torch.matmul(C_real.t(), C_pred)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))
    trace_term = torch.trace(cov_pred + cov_real) - 2.0 * sq_tr_cov  # scalar

    # |mu_real - mu_pred|^2
    diff = mu_real - mu_pred  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()
