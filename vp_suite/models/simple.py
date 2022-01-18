import torch
from torch import nn as nn
from vp_suite.models._base_model import VideoPredictionModel


class SimpleV1(VideoPredictionModel):

    # model-specific constants
    NAME = "SimpleV1"
    REQUIRED_ARGS = ["img_shape", "action_size", "tensor_value_range", "temporal_dim"]

    # model hyperparameters
    temporal_dim = None

    def __init__(self, device, **model_args):
        super(SimpleV1, self).__init__(device, **model_args)

        self.min_context_frames = self.temporal_dim
        self.act_fn = nn.ReLU(inplace=True)
        self.cnn = nn.Conv3d(self.img_c, self.img_c, kernel_size=(self.temporal_dim, 5, 5),
                             stride=(1, 1, 1), padding=(0, 2, 2), bias=False)
        self._init_to_clf()

    def _init_to_clf(self):
        clf_weights = torch.zeros(self.img_c, self.img_c, self.temporal_dim, 5, 5)
        for i in range(self.img_c):
            clf_weights[i, i, -1, 2, 2] = 1.0
        self.cnn.weight.data = clf_weights

    def _config(self):
        return {"temporal_dim": self.temporal_dim}

    def pred_1(self, x, **kwargs):
        assert x.shape[2] == self.temporal_dim, "invalid number of frames given"
        return self.cnn(x).squeeze(dim=2)  # [b, c, h, w]

    def forward(self, x, pred_length=1, **kwargs):
        x = x.transpose(1, 2)  # shape: [b, c, t, h, w]
        output_frames = []
        for t in range(pred_length):
            input = x[:, :, -self.temporal_dim:]
            output_frames.append(self.pred_1(input))
        return torch.stack(output_frames, dim=1), None


class SimpleV2(VideoPredictionModel):

    # model-specific constants
    NAME = "SimpleV2"
    REQUIRED_ARGS = ["img_shape", "action_size", "tensor_value_range", "temporal_dim"]

    # model hyperparameters
    hidden_channels = 64
    temporal_dim = None

    def _config(self):
        return {
            "temporal_dim": self.temporal_dim,
            "hidden_channels": self.hidden_channels
        }

    def __init__(self, device, **model_args):
        super(SimpleV2, self).__init__(device, **model_args)

        self.min_context_frames = self.temporal_dim
        self.act_fn = nn.ReLU(inplace=True)
        self.big_branch = nn.Sequential(
            nn.Conv3d(self.img_c, self.hidden_channels, (self.temporal_dim, 5, 5), (1, 1, 1), (0, 2, 2), bias=False),
            nn.Conv3d(self.hidden_channels, self.hidden_channels, (1, 5, 5), (1, 1, 1), (0, 2, 2), bias=False),
            nn.Conv3d(self.hidden_channels, self.img_c, (1, 5, 5), (1, 1, 1), (0, 2, 2), bias=False)

        )
        self.final_merge = nn.Conv2d(2 * self.img_c, self.img_c, kernel_size=(5, 5), stride=(1, 1),
                                     padding=(2, 2), bias=False)
        self.init_to_clf()

    def init_to_clf(self):
        clf_weights = torch.zeros(self.img_c, 2 * self.img_c, 5, 5)
        for i in range(self.img_c):
            clf_weights[i, self.img_c + i, 2, 2] = 1.0
        self.final_merge.weight.data = clf_weights

    def pred_1(self, x, **kwargs):
        assert x.shape[2] == self.temporal_dim, "invalid number of frames given"
        last_frame = x[:, :, -1]  # [b, c, h, w]
        big_branch = self.big_branch(x).squeeze(2)  # [b, c, h, w]
        out = self.final_merge(torch.cat([big_branch, last_frame], dim=1))
        return out

    def forward(self, x, pred_length=1, **kwargs):
        x = x.transpose(1, 2)  # shape: [b, c, t, h, w]
        output_frames = []
        for t in range(pred_length):
            input = x[:, :, -self.temporal_dim:]
            output_frames.append(self.pred_1(input))
        return torch.stack(output_frames, dim=1), None
