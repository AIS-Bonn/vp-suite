import torch
from torch import nn as nn
import sys
sys.path.append("")
from vp_suite.models.base_model import VideoPredictionModel


class SimpleV1(VideoPredictionModel):
    def __init__(self, in_channels, out_channels, time_slice, action_size, device):
        super(SimpleV1, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels, "model only works for equal in and out channels"
        self.time_slice = time_slice
        self.action_size = action_size
        self.use_actions = action_size > 0
        self.act_fn = nn.ReLU(inplace=True)
        self.cnn = nn.Conv3d(self.in_channels, self.out_channels, (self.time_slice, 5, 5), (1, 1, 1), (0, 2, 2), bias=False)
        self.init_to_clf()
        self.to(device)

    def init_to_clf(self):
        clf_weights = torch.zeros(self.in_channels, self.in_channels, self.time_slice, 5, 5)
        for i in range(self.in_channels):
            clf_weights[i, i, -1, 2, 2] = 1.0
        self.cnn.weight.data = clf_weights

    def forward(self, x, **kwargs):
        assert x.shape[2] == self.time_slice, "invalid number of frames given"
        return self.cnn(x)

    def pred_n(self, x, pred_length=1, **kwargs):
        x = x.transpose(1, 2)  # shape: [b, c, t, h, w]
        output_frames = []
        for t in range(pred_length):
            input = x[:, :, -self.time_slice:]
            output_frames.append(self.forward(input))
        return torch.cat(output_frames, dim=2).transpose(1, 2), None


class SimpleV2(VideoPredictionModel):

    hidden_channels = 64

    def __init__(self, in_channels, out_channels, time_slice, action_size, device):
        super(SimpleV2, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels, "model only works for equal in and out channels"
        self.time_slice = time_slice
        self.action_size = action_size
        self.use_actions = action_size > 0
        self.act_fn = nn.ReLU(inplace=True)
        self.big_branch = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_channels, (self.time_slice, 5, 5), (1, 1, 1), (0, 2, 2), bias=False),
            nn.Conv3d(self.hidden_channels, self.hidden_channels, (1, 5, 5), (1, 1, 1), (0, 2, 2), bias=False),
            nn.Conv3d(self.hidden_channels, self.out_channels, (1, 5, 5), (1, 1, 1), (0, 2, 2), bias=False)

        )
        self.final_merge = nn.Conv2d(2 * self.in_channels, self.out_channels, kernel_size=(5, 5), stride=(1, 1),
                                     padding=(2, 2), bias=False)
        self.init_to_clf()
        self.to(device)

    def init_to_clf(self):
        clf_weights = torch.zeros(self.out_channels, 2 * self.in_channels, 5, 5)
        for i in range(self.out_channels):
            clf_weights[i, self.in_channels + i, 2, 2] = 1.0
        self.final_merge.weight.data = clf_weights

    def forward(self, x, **kwargs):
        assert x.shape[2] == self.time_slice, "invalid number of frames given"
        last_frame = x[:, :, -1]  # [b, c, h, w]
        big_branch = self.big_branch(x).squeeze(2)  # [b, c, h, w]
        out = self.final_merge(torch.cat([big_branch, last_frame], dim=1))
        return out

    def pred_n(self, x, pred_length=1, **kwargs):
        x = x.transpose(1, 2)  # shape: [b, c, t, h, w]
        output_frames = []
        for t in range(pred_length):
            input = x[:, :, -self.time_slice:]
            output_frames.append(self.forward(input))
        return torch.stack(output_frames, dim=1), None

if __name__ == '__main__':
    simple_model = SimpleV2(3, 3, 5, 0, "cuda")
    print(sum(p.numel() for p in simple_model.parameters() if p.requires_grad))
    T_in = 7
    T_pred = 3
    x = torch.randn(1, T_in, 3, 3, 3, device="cuda")
    x_pred, _ = simple_model.pred_n(x, T_pred)
    x_last = x[:, -1]
    for i in range(1, T_pred+1):
        assert torch.equal(x_last[..., 2:-2, 2:-2], x_pred[:, -i, :, 2:-2, 2:-2])
