import math
import os
from pathlib import Path

from torch import linalg as linalg

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from vp_suite.measure.fvd.pytorch_i3d.pytorch_i3d import InceptionI3d
from vp_suite.measure._base_measure import BaseMeasure

class FrechetVideoDistance(BaseMeasure):
    """INSPIRED BY: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

    :param a: b, defaults to c
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    '''
    '''

    min_T = 9
    max_T = 16
    i3d_in_shape = (224, 224)
    i3d_num_classes = 400
    i3d_ckpt_file = str((Path(__file__).parent / "pytorch_i3d" / "models" / "rgb_imagenet.pt").resolve())
    input_chunks = 1
    drop_last_chunk = False

    def __init__(self, device, in_channels=3):
        super(FrechetVideoDistance, self).__init__(device)

        self.i3d = InceptionI3d(num_classes=self.i3d_num_classes, in_channels=in_channels)
        self.i3d.load_state_dict(torch.load(Path(__file__).parent / self.i3d_ckpt_file))
        self.i3d.to(self.device)
        self.i3d.eval()  # don't train the pre-trained I3D
        self.to(self.device)

    @classmethod
    def valid_T(cls, x):
        return x >= cls.min_T and x <= cls.max_T

    def valid_frame_count(self, num_frames):
        ok = True
        if num_frames < self.min_T:
            print(f"The I3D Module used for FVD needs at least"
                             f" 9 input frames (given: {num_frames}) -> returning None as loss value!")
            ok = False
        elif num_frames > self.max_T:
            self.determine_number_of_chunks(num_frames)
            print(f"Warning: The I3D Module used for FVD handles at most 16 input frames (given: {num_frames})"
                  f" -> input video will be chunked into {self.input_chunks} chunks!")
            ok = True
        return ok

    def determine_number_of_chunks(self, n):
        '''
        If given input length is too large, this function returns the number of chunks.
        Each chunk is then used for a separate fvd calculation, and their results are combined afterwards.
        '''
        possible_chunk_l = range(self.max_T, self.min_T-1, -1)
        for chunk_l in possible_chunk_l:
            if n % chunk_l >= self.min_T:
                self.input_chunks = n // chunk_l + 1

        # loss-less chunking not possible -> get largest possible even chunk and drop last chunk
        if self.input_chunks == None:
            missed_frames = [n % chunk_l for chunk_l in possible_chunk_l]
            best_chunk_l = sorted(zip(possible_chunk_l, missed_frames), key=lambda x: x[1])[-1]
            self.drop_last_chunk = True
            self.input_chunks = n // best_chunk_l + 1


    def forward(self, pred, real):
        # input: [b, T, c, h, w]
        # output: scalar

        vid_shape = pred.shape
        if vid_shape != real.shape:
            raise ValueError("FrechetVideoDistance.get_distance(pred, real): vid shapes not equal!")

        num_frames = vid_shape[1]
        if not self.valid_frame_count(num_frames):
            return None

        # resize images in video to 224x224 because the I3D network needs that
        pred = TF.resize(pred.reshape(-1, *vid_shape[2:]), self.i3d_in_shape)
        real = TF.resize(real.reshape(-1, *vid_shape[2:]), self.i3d_in_shape)

        # re-arrange dims for I3D input
        pred = pred.reshape(*vid_shape[:3], *self.i3d_in_shape).permute((0, 2, 1, 3, 4))  # [b, c, T, 224, 224]
        real = real.reshape(*vid_shape[:3], *self.i3d_in_shape).permute((0, 2, 1, 3, 4))

        pred_chunked = torch.chunk(pred, self.input_chunks, dim=2)
        real_chunked = torch.chunk(real, self.input_chunks, dim=2)

        n_valid_chunks = self.input_chunks if not self.drop_last_chunk else self.input_chunks - 1
        chunk_distances = [self.get_distance(pred_chunked[i], real_chunked[i]) for i in range(n_valid_chunks)]
        return sum(chunk_distances) / n_valid_chunks  # mean

    def get_distance(self, pred, real):

        # input: [b, c, t, 224, 224] with t in suitable (chunked) size
        # output: scalar

        logits_pred = self.i3d.extract_features(pred).squeeze()  # [b, n]
        logits_real = self.i3d.extract_features(real).squeeze()

        if pred.shape[0] == 1:  # if batch size is 1, the prev. squeeze also removed the batch dim
            logits_pred = logits_pred.unsqueeze(dim=0)
            logits_real = logits_real.unsqueeze(dim=0)

        return calculate_2_wasserstein_dist(logits_pred, logits_real)


def calculate_2_wasserstein_dist(pred, real):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_real, P_pred = min_{X, Y} E[|X-Y|^2]

    For multivariate gaussian distributed inputs x_real ~ MN(mu_real, cov_real) and x_pred ~ MN(mu_pred, cov_pred),
    this reduces to: d = |mu_real - mu_pred|^2 - Tr(cov_real + cov_pred - 2(cov_real * cov_pred)^(1/2))

    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf

    Input shape: [b = batch_size, n = num_features]
    Output shape: scalar
    '''

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



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b, T, c, h, w = 5, 9, 3, 270, 480
    a, b = torch.randn((b, T, c, h, w), device=device), torch.randn((b, T, c, h, w), device=device)
    fvd = FrechetVideoDistance(device=device, num_frames=T, in_channels=c)
    loss = fvd.get_distance(a, b)
    print(loss.item())