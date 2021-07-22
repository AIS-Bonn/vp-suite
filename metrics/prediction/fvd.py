import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from models.i3d.pytorch_i3d import InceptionI3d
from config import DEVICE
from utils import get_2_wasserstein_dist


class FrechetVideoDistance(nn.Module):
    '''
    INSPIRED BY: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
    '''

    def __init__(self, num_frames, in_channels=3):
        super(FrechetVideoDistance, self).__init__()

        if num_frames < 9:
            raise ValueError(f"The I3D Module used for FVD needs at least 9 input frames (given: {num_frames})!")

        self.i3d_in_shape = (224, 224)

        self.i3d = InceptionI3d(num_classes=400, in_channels=in_channels)
        self.i3d.load_state_dict(torch.load("models/i3d/models/rgb_imagenet.pt"))
        self.i3d.to(DEVICE)
        self.i3d.eval()  # don't train the pre-trained I3D


    def forward(self, pred, real):

        # input: [b, T, c, h, w]
        # output: scalar

        vid_shape = pred.shape
        if vid_shape != real.shape:
            raise ValueError("FrechetVideoDistance.get_distance(pred, real): vid shapes not equal!")

        # resize images in video to 224x224 because the I3D network needs that
        pred = TF.resize(pred.reshape(-1, *vid_shape[2:]), self.i3d_in_shape)
        real = TF.resize(real.reshape(-1, *vid_shape[2:]), self.i3d_in_shape)

        # re-arrange dims for I3D input
        pred = pred.reshape(*vid_shape[:3], *self.i3d_in_shape).permute((0, 2, 1, 3, 4))  # [b, T, c, 224, 224]
        real = real.reshape(*vid_shape[:3], *self.i3d_in_shape).permute((0, 2, 1, 3, 4))

        logits_pred = self.i3d.extract_features(pred).squeeze()  # [b, n]
        logits_real = self.i3d.extract_features(real).squeeze()

        if vid_shape[0] == 1:  # if batch size is 1, the prev. squeeze also removed the batch dim
            logits_pred = logits_pred.unsqueeze(dim=0)
            logits_real = logits_real.unsqueeze(dim=0)

        return get_2_wasserstein_dist(logits_pred, logits_real)


if __name__ == '__main__':
    b, T, c, h, w = 5, 9, 3, 270, 480
    a, b = torch.randn((b, T, c, h, w)).to(DEVICE), torch.randn((b, T, c, h, w)).to(DEVICE)
    fvd = FrechetVideoDistance(num_frames=T)
    loss = fvd.get_distance(a, b)
    print(loss.item())