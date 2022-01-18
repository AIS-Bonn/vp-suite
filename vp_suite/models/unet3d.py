import torch
from torch import nn as nn
from torchvision.transforms import functional as TF

from vp_suite.models.model_blocks.conv import DoubleConv3d, DoubleConv2d
from vp_suite.models._base_model import VideoPredictionModel


class UNet3D(VideoPredictionModel):

    # model-specific constants
    NAME = "UNet-3D"
    REQUIRED_ARGS = ["img_shape", "action_size", "tensor_value_range", "temporal_dim"]
    CAN_HANDLE_ACTIONS = True

    # model hyperparameters
    features = [8, 16, 32, 64]
    temporal_dim = None

    def _config(self):
        return {
            "temporal_dim": self.temporal_dim,
            "features": self.features
        }

    def __init__(self, device, **model_args):
        super(UNet3D, self).__init__(device, **model_args)

        self.min_context_frames = self.temporal_dim
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.time3ds = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        if self.action_conditional:
            self.action_inflates = nn.ModuleList()

        # down
        cur_in_channels = self.img_c
        cur_img_h, cur_img_w = self.img_h, self.img_w
        for feature in self.features:
            if self.action_conditional:
                self.action_inflates.append(nn.Linear(in_features=self.action_size,
                                                      out_features=self.action_size * cur_img_h * cur_img_w))
                zeros = torch.zeros(1, 1, cur_img_h, cur_img_w)
                pooled_zeros = self.pool(zeros)
                cur_img_h, cur_img_w = pooled_zeros.shape[-2:]
                cur_in_channels += self.action_size
            self.downs.append(DoubleConv3d(in_c=cur_in_channels, out_c=feature))
            self.time3ds.append(nn.Conv3d(in_channels=feature, out_channels=feature, kernel_size=(self.temporal_dim, 1, 1)))
            cur_in_channels = feature

        # bottleneck
        bn_h, bn_w = cur_img_h, cur_img_w
        bn_feat = self.features[-1]
        self.time3ds.append(nn.Conv3d(in_channels=bn_feat, out_channels=bn_feat, kernel_size=(self.temporal_dim, 1, 1)))
        if self.action_conditional:
            self.bottleneck_action_inflate = nn.Linear(in_features=self.action_size,
                                                       out_features=self.action_size * bn_h * bn_w)
            self.bottleneck = DoubleConv2d(in_c=bn_feat + self.action_size, out_c=bn_feat * 2)
        else:
            self.bottleneck = DoubleConv2d(in_c=bn_feat, out_c=bn_feat * 2)

        # up
        for feature in reversed(self.features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature,
                                               kernel_size=(2, 2), stride=(2, 2)))
            self.ups.append(DoubleConv2d(in_c=feature * 2, out_c=feature))

        # final
        self.final_conv = nn.Conv2d(in_channels=self.features[0], out_channels=self.img_c, kernel_size=(1, 1))

    def forward(self, x, pred_length=1, **kwargs):
        # input: T_in frames: [b, T_in, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]
        preds = []

        # actions
        b, input_length, _, _, _ = x.shape
        empty_actions = torch.zeros(b, input_length + pred_length, device=self.device)
        actions = kwargs.get("actions", empty_actions)


        for t in range(pred_length):
            pred = self.pred_1(x, actions=actions)
            pred = pred.unsqueeze(dim=1)
            preds.append(pred)
            x = torch.cat([x[:, 1:], pred], dim=1)

        pred = torch.cat(preds, dim=1)
        return pred, None

    def pred_1(self, x, **kwargs):
        # input: T frames: [b, T, c, h, w] and T actions: [b, T, a]
        # output: single frame: [b, c, h, w]
        T_in = x.shape[1]
        x = x[:, -self.temporal_dim:].permute((0, 2, 1, 3, 4))  # [b, c, temporal_dim, h, w]
        actions = kwargs.get("actions", torch.zeros(1, 1))
        if self.action_conditional:
            if actions.ndim != 3 or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
        actions = actions[:, T_in-self.temporal_dim:T_in].transpose(0, 1)

        # DOWN
        skip_connections = []
        for i in range(len(self.downs)):
            if self.action_conditional:
                actions_ = actions.clone()
                actions_ = actions_.reshape(-1, self.action_size)  # [temporal_dim*b, a]
                inflated_action = self.action_inflates[i](actions_).view(-1, self.action_size, *x.shape[-2:])  # [temporal_dim*b, a, h, w]
                inflated_action = inflated_action.reshape(*actions.shape[:2], *inflated_action.shape[1:])  # [temporal_dim, b, a, h, w]
                inflated_action = inflated_action.permute((1, 2, 0, 3, 4))  # [b, a, temporal_dim, h, w]
                x = torch.cat([x, inflated_action], dim=1)
            x = self.downs[i](x)

            skip_connection = self.time3ds[i](x).squeeze(dim=2)
            skip_connections.append(skip_connection)
            x = self.pool(x)

        x = self.time3ds[-1](x).squeeze(dim=2)  # from [b, feat[-1], temporal_dim, h, w] to [b, feat[-1], h, w]
        if self.action_conditional:
            last_action = actions[-1]  # [b, a]
            inflated_action = self.bottleneck_action_inflate(last_action)  # [b, a*h*w]
            inflated_action = inflated_action.view(-1, self.action_size, *x.shape[-2:])  # [b, a, h, w]
            x = torch.cat([x, inflated_action], dim=1)
        x = self.bottleneck(x)

        # UP
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        # FINAL
        return self.final_conv(x)