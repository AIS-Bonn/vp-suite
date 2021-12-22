import torch
from torch import nn as nn
from torchvision.transforms import functional as TF

from vp_suite.models.model_blocks.conv import DoubleConv3d, DoubleConv2d
from vp_suite.models.base_model import VideoPredictionModel


class UNet3dModel(VideoPredictionModel):

    features = [8, 16, 32, 64]
    time_dim = None  # if None, use all context frames
    can_handle_actions = True

    @classmethod
    def model_desc(cls):
        return "UNet-3D"

    def __init__(self, cfg):
        super(UNet3dModel, self).__init__(cfg)

        self.time_dim = self.time_dim or cfg.context_frames
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.time3ds = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        if self.use_actions:
            self.action_inflates = nn.ModuleList()

        # down
        cur_in_channels = self.img_c
        cur_img_h, cur_img_w = self.img_h, self.img_w
        for feature in self.features:
            if self.use_actions:
                self.action_inflates.append(nn.Linear(in_features=self.action_size,
                                                      out_features=self.action_size * cur_img_h * cur_img_w))
                zeros = torch.zeros(1, 1, cur_img_h, cur_img_w)
                pooled_zeros = self.pool(zeros)
                cur_img_h, cur_img_w = pooled_zeros.shape[-2:]
                cur_in_channels += self.action_size
            self.downs.append(DoubleConv3d(in_c=cur_in_channels, out_c=feature))
            self.time3ds.append(nn.Conv3d(in_channels=feature, out_channels=feature, kernel_size=(self.time_dim, 1, 1)))
            cur_in_channels = feature

        # bottleneck
        bn_h, bn_w = cur_img_h, cur_img_w
        bn_feat = self.features[-1]
        self.time3ds.append(nn.Conv3d(in_channels=bn_feat, out_channels=bn_feat, kernel_size=(self.time_dim, 1, 1)))
        if self.use_actions:
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

    def pred_n(self, x, pred_length=1, **kwargs):
        # input: T_in frames: [b, T_in, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]
        preds = []
        loss_dicts = []
        input_length = x.shape[1]

        # actions
        actions = kwargs.get("actions", [None] * (input_length + pred_length))
        if self.use_actions:
            if actions is None or actions[0] == None or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
        if type(actions) == torch.Tensor:
                actions = actions.transpose(0, 1)  # [T_in+pred, b, ...]

        for t in range(pred_length):
            pred, loss_dict = self.forward(x, **{"actions": actions[t:t+input_length]})
            pred = pred.unsqueeze(dim=1)
            preds.append(pred)
            loss_dicts.append(loss_dict)
            x = torch.cat([x[:, 1:], pred], dim=1)

        pred = torch.cat(preds, dim=1)
        if loss_dicts[0] is not None:
            loss_dict = {k: torch.mean([loss_dict[k] for loss_dict in loss_dicts]) for k in loss_dicts[0]}
        else:
            loss_dict = None
        return pred, loss_dict

    def forward(self, x, **kwargs):
        # input: T frames: [b, T, c, h, w]
        # output: single frame: [b, c, h, w]
        assert x.shape[1] == self.time_dim, f"{self.time_dim} frames needed as pred input, {x.shape[1]} are given"
        x = x.permute((0, 2, 1, 3, 4))  # [b, c, T, h, w]
        b, _, T, _, _ = x.shape
        skip_connections = []
        actions = kwargs.get("actions", None)  # [T, b, a]
        if self.use_actions:
            assert actions.shape[-1] == self.action_size, "action size mismatch"

        # DOWN
        for i in range(len(self.downs)):
            if self.use_actions:
                actions_ = actions.clone()
                actions_ = actions_.reshape(-1, self.action_size)  # [T*b, a]
                inflated_action = self.action_inflates[i](actions_).view(-1, self.action_size, *x.shape[-2:])  # [T*b, a, h, w]
                inflated_action = inflated_action.reshape(T, b, *inflated_action.shape[1:])  # [T, b, a, h, w]
                inflated_action = inflated_action.permute((1, 2, 0, 3, 4))  # [b, a, T, h, w]
                x = torch.cat([x, inflated_action], dim=1)
            x = self.downs[i](x)

            skip_connection = self.time3ds[i](x).squeeze(dim=2)
            skip_connections.append(skip_connection)
            x = self.pool(x)

        x = self.time3ds[-1](x).squeeze(dim=2)  # from [b, feat[-1], T, h, w] to [b, feat[-1], h, w]
        if self.use_actions:
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
        return self.final_conv(x), None