import torch
from torch import nn as nn
import sys
sys.path.append("")
from vp_suite.models._base_model import VideoPredictionModel


class SimpleV1(VideoPredictionModel):

    min_context_frames = 5
    temporal_dim = min_context_frames

    @classmethod
    def model_desc(cls):
        return "SimpleV1"

    def __init__(self, trainer_cfg, **model_args):
        super(SimpleV1, self).__init__(trainer_cfg)

        self.act_fn = nn.ReLU(inplace=True)
        self.cnn = nn.Conv3d(self.img_c, self.img_c, kernel_size=(self.temporal_dim, 5, 5),
                             stride=(1, 1, 1), padding=(0, 2, 2), bias=False)
        self.init_to_clf()

    def init_to_clf(self):
        clf_weights = torch.zeros(self.img_c, self.img_c, self.temporal_dim, 5, 5)
        for i in range(self.img_c):
            clf_weights[i, i, -1, 2, 2] = 1.0
        self.cnn.weight.data = clf_weights

    def forward(self, x, **kwargs):
        assert x.shape[2] == self.temporal_dim, "invalid number of frames given"
        return self.cnn(x)

    def pred_n(self, x, pred_length=1, **kwargs):
        x = x.transpose(1, 2)  # shape: [b, c, t, h, w]
        output_frames = []
        for t in range(pred_length):
            input = x[:, :, -self.temporal_dim:]
            output_frames.append(self.forward(input))
        return torch.cat(output_frames, dim=2).transpose(1, 2), None


class SimpleV2(VideoPredictionModel):

    hidden_channels = 64
    min_context_frames = 5
    temporal_dim = min_context_frames

    @classmethod
    def model_desc(cls):
        return "SimpleV2"

    def __init__(self, trainer_cfg, **model_args):
        super(SimpleV2, self).__init__(trainer_cfg)

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

    def forward(self, x, **kwargs):
        assert x.shape[2] == self.temporal_dim, "invalid number of frames given"
        last_frame = x[:, :, -1]  # [b, c, h, w]
        big_branch = self.big_branch(x).squeeze(2)  # [b, c, h, w]
        out = self.final_merge(torch.cat([big_branch, last_frame], dim=1))
        return out

    def pred_n(self, x, pred_length=1, **kwargs):
        x = x.transpose(1, 2)  # shape: [b, c, t, h, w]
        output_frames = []
        for t in range(pred_length):
            input = x[:, :, -self.temporal_dim:]
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
