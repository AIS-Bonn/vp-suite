import torch
import torch.nn as nn
import torch.nn.functional as F

from models.i3d.pytorch_i3d import InceptionI3d
from config import DEVICE


class FrechetVideoDistance(nn.Module):

    def __init__(self):
        super(FrechetVideoDistance, self).__init__()

        self.i3d = InceptionI3d(num_classes=400, in_channels=3)
        self.i3d.replace_logits(157)
        self.i3d.load_state_dict(torch.load("models/i3d/models/rgb_imagenet.pt"))
        self.i3d.to(DEVICE)
        self.i3d.eval()


    def get_distance(self, pred, real):
        logits_pred, _ = self.i3d.extract_features(pred)
        logits_real, _ = self.i3d.extract_features(real)

        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
        mean_pred = torch.mean(logits_pred, dim=0)
        mean_real = torch.mean(logits_real, dim=0)
        n = mean_pred.shape[0]

        covar_pred = torch.bmm(mean_pred, mean_pred.transpose(-1, -2)).div(n-1)
        covar_real = torch.bmm(mean_real, mean_real.transpose(-1, -2)).div(n-1)


        # Compute the two components of FID.

        # First the covariance component.
        sqrt_trace_component = torch.trace(torch.sqrt(torch.bmm(covar_real, covar_pred)))
        trace_term = torch.trace(covar_real + covar_pred) - 2.0 * sqrt_trace_component

        # Next the distance between means.
        diff = mean_real - mean_pred
        mean_term = torch.sum(torch.mul(diff, diff))

        return trace_term + mean_term


if __name__ == '__main__':
    raise NotImplementedError
    # TODO implement FVD testing