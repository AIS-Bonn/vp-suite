import math
from pathlib import Path

import torch
from torch import linalg as linalg
import torchvision.transforms.functional as TF

from vp_suite.measure.fvd._pytorch_i3d.pytorch_i3d import InceptionI3d
from vp_suite.base import VPMeasure


class FrechetVideoDistance(VPMeasure):
    r"""
    This measure calculates the Frechet Video Distance, as introduced in Unterthiner et al.
    (https://arxiv.org/abs/1812.01717). The Frechet Distance is a similarity measure between two curves,
    and the Frechet Video Distance transfers this idea to assess the perceptual quality of generated videos with
    respect to a ground truth sequence by comparing video features obtained by passing the videos to an InceptionI3D
    Network.

    Code is inspired by: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

    Note:
        The Frechet Video Distance calculation code is differentiable, meaning that this version of FVD can also be
        used as a loss!
    """
    NAME = "Fréchet Video Distance (FVD)"
    REFERENCE = "https://arxiv.org/abs/1812.01717"

    _MIN_T = 9  #: The minimum number of frames per sequence needed for FVD calculation.
    _MAX_T = 16  #: The maximum number of framed per sequence usable for FVD calculation in a singe chunk.
    _I3D_IN_SIZE = (224, 224)  #: The expected frame dimensions of the I3D Network.
    _I3D_NUM_CLASSES = 400  #: The number of classes (vector dimensionality) the I3D Network returns.
    _I3D_CKPT_FILE = "_pytorch_i3d/models/rgb_imagenet.pt"  #: The file path to the pretrained I3D Model

    def __init__(self, device, in_channels=3):
        r"""
        Instantiates the FVD by setting the device and initializing the InceptionI3D module, which is used to
        extract the features that shall be compared.

        Args:
            device (str): A string specifying whether to use the GPU for calculations (`cuda`) or the CPU (`cpu`).
            in_channels (int): Number of input channels (Supported: 2 or 3)
        """
        super(FrechetVideoDistance, self).__init__(device)

        self.i3d = InceptionI3d(num_classes=self._I3D_NUM_CLASSES, in_channels=in_channels)
        self.i3d.load_state_dict(torch.load(Path(__file__).parent / self._I3D_CKPT_FILE))
        self.i3d.to(self.device)
        self.i3d.eval()  # don't train the pre-trained I3D
        self.to(self.device)

    def calculate_n_chunks(self, num_frames):
        r"""
        If given input length is too large, this function returns the number of chunks.
        Each chunk is then used for a separate fvd calculation, and their results are combined afterwards.

        Args:
            num_frames (int): The number of context frames (aka the input length).

        Returns: 
            The number of chunks the input sequence needs to be split into, 
            as well as a boolean value indicating whether the last chunk has to be neglected.

        """
        n_chunks, drop_last_chunk = 1, False

        if num_frames < self._MIN_T:
            print(f"The I3D Module used for FVD needs at least"
                  f" {self._MIN_T} input frames (given: {num_frames}) -> returning None as loss value!")
            n_chunks = -1

        elif num_frames > self._MAX_T:
            possible_chunk_l = range(self._MAX_T, self._MIN_T - 1, -1)
            n_chunks = None
            for chunk_l in possible_chunk_l:
                if num_frames % chunk_l >= self._MIN_T:
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

    def forward(self, pred, target):
        vid_shape = pred.shape
        if vid_shape != target.shape:
            raise ValueError("FrechetVideoDistance.get_distance(pred, target): vid shapes not equal!")

        # determine whether input needs to be chunked for processing
        num_frames = vid_shape[1]
        n_chunks, drop_last_chunk = self.calculate_n_chunks(num_frames)
        if n_chunks < 1:
            return None

        # resize images in video to 224x224 because the I3D network needs that
        pred = TF.resize(pred.reshape(-1, *vid_shape[2:]), self._I3D_IN_SIZE)
        target = TF.resize(target.reshape(-1, *vid_shape[2:]), self._I3D_IN_SIZE)

        # re-arrange dims for I3D input
        pred = pred.reshape(*vid_shape[:3], *self._I3D_IN_SIZE).permute((0, 2, 1, 3, 4))  # [b, c, T, 224, 224]
        target = target.reshape(*vid_shape[:3], *self._I3D_IN_SIZE).permute((0, 2, 1, 3, 4))

        pred_chunked = torch.chunk(pred, n_chunks, dim=2)
        target_chunked = torch.chunk(target, n_chunks, dim=2)

        n_valid_chunks = (n_chunks - 1) if drop_last_chunk else n_chunks
        chunk_distances = [self.get_distance(pred_chunked[i], target_chunked[i]) for i in range(n_valid_chunks)]
        return sum(chunk_distances) / n_valid_chunks  # mean of chunks

    def get_distance(self, pred, target):
        r"""
        Calculates the Frechet Video Distance between the provided chunked prediction and the ground truth tensors,
        by first extracting perceptual features from the InceptionI3D Network before calculating the
        2-Wasserstein-Distance on these features. The video frames have been previously resized to meet the height and
        width constraints of the I3D Network.

        Args:
            pred (torch.Tensor): The chunked predicted frame sequence as a 5D tensor (batch, c, chunk_l, h, w).
            target (torch.Tensor): The chunked ground truth frame sequence as a 5D tensor (batch, c, chunk_l, h, w).

        Returns: The calculated 2-Wasserstein metric as a scalar tensor.
        """
        logits_pred = self.i3d.extract_features(pred).squeeze()  # [b, n]
        logits_target = self.i3d.extract_features(target).squeeze()

        if pred.shape[0] == 1:  # if batch size is 1, the prev. squeeze also removed the batch dim
            logits_pred = logits_pred.unsqueeze(dim=0)
            logits_target = logits_target.unsqueeze(dim=0)

        return calculate_2_wasserstein_dist(logits_pred, logits_target)


def calculate_2_wasserstein_dist(pred, target):
    r"""
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: ``d(P_target, P_pred = min_{X, Y} E[|X-Y|^2]``

    For multivariate gaussian distributed inputs ``x_target ~ MN(mu_target, cov_target)``
    and ``x_pred ~ MN(mu_pred, cov_pred)``, this reduces to:
    ``d = |mu_target - mu_pred|^2 - Tr(cov_target + cov_pred - 2(cov_target * cov_pred)^(1/2))``

    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf

    Input shape: [b = batch_size, n = num_features]
    Output shape: scalar

    Args:
        pred (torch.Tensor): The logits of the chunked prediction extracted by the InceptionI3D network (batch, n_feat).
        target (torch.Tensor): The logits of the chunked ground truth extracted by the I3D network (batch, n_feat).

    Returns: The calculated 2-Wasserstein metric as a scalar tensor.
    """
    if pred.shape != target.shape:
        raise ValueError("Expecting equal shapes for pred and target!")

    # the following ops need some extra precision
    pred, target = pred.transpose(0, 1).double(), target.transpose(0, 1).double()  # [n, b]
    mu_pred, mu_target = torch.mean(pred, dim=1, keepdim=True), torch.mean(target, dim=1, keepdim=True)  # [n, 1]
    n, b = pred.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_pred = pred - mu_pred
    E_target = target - mu_target
    cov_pred = torch.matmul(E_pred, E_pred.t()) * fact  # [n, n]
    cov_target = torch.matmul(E_target, E_target.t()) * fact

    # calculate Tr((cov_target * cov_pred)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues of the mm(cov_pred, cov_target) and therefore for M are target-valued.
    C_pred = E_pred * math.sqrt(fact)  # [n, n], "root" of covariance
    C_target = E_target * math.sqrt(fact)
    M_l = torch.matmul(C_pred.t(), C_target)
    M_r = torch.matmul(C_target.t(), C_pred)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_target + cov_pred - 2(cov_target * cov_pred)^(1/2))
    trace_term = torch.trace(cov_pred + cov_target) - 2.0 * sq_tr_cov  # scalar

    # |mu_target - mu_pred|^2
    diff = mu_target - mu_pred  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()
