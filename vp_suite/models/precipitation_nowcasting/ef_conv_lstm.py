from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from vp_suite.base.base_model import VideoPredictionModel
from vp_suite.models.model_blocks.convlstm_hzzone.conv_lstm import ConvLSTM
from vp_suite.models.precipitation_nowcasting.ef_blocks import Encoder, Forecaster


class EF_ConvLSTM(VideoPredictionModel):
    r"""
    This is a reimplementation of the Encoder-Forecaster model based on ConvLSTMs, as introduced in
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.
    (https://arxiv.org/abs/1506.04214). This implementation is based on the PyTorch implementation on
    https://github.com/Hzzone/Precipitation-Nowcasting.

    The Encoder-Forecaster Network stacks multiple ConvLSTM layers that operate on different spatial scales.

    Note:
        In its basic version, this model predicts as many frames as it got fed (t_pred == t_in).
        For t_pred < t_in, the same procedure will be used and excess frames will be discarded.
        For t_pred > t_in, multiple consecutive forward passes will be combined that each work on the previous result.
    """

    # model-specific constants
    NAME = "EF-ConvLSTM (Shi et al.)"

    # model hyperparameters
    num_layers = 3
    enc_names = ["conv1_leaky_1", "conv2_leaky_1", "conv3_leaky_1"]
    dec_names = ["deconv1_leaky_1", "deconv2_leaky_1", "deconv3_leaky_1"]
    final_conv_1_name = ["conv3_leaky_2"]
    final_conv_2_name = ["conv3_3"]
    enc_c = [8, 64, 192, 192, 192, 192]  #: Channels; Length should be 2*num_layers
    dec_c = [192, 192, 192, 64, 64, 8]  #: Channels; Length should be 2*num_layers
    final_conv_1_c = 8
    enc_rnn_state_h = [96, 32, 16]
    enc_rnn_state_w = [96, 32, 16]
    dec_rnn_state_h = [16, 32, 96]
    dec_rnn_state_w = [16, 32, 96]
    enc_k = [7, 3, 5, 3, 3, 3]  #: Kernel sizes; Length should be 2*num_layers
    dec_k = [3, 4, 3, 5, 3, 7]  #: Kernel sizes; Length should be 2*num_layers
    final_conv_1_k = 3
    final_conv_2_k = 1
    enc_s = [5, 1, 3, 1, 2, 1]  #: Strides; Length should be 2*num_layers
    dec_s = [1, 2, 1, 3, 1, 5]  #: Strides; Length should be 2*num_layers
    final_conv_1_s = 1
    final_conv_2_s = 1
    enc_p = [1, 1, 1, 1, 1, 1]  #: Paddings; Length should be 2*num_layers
    dec_p = [1, 1, 1, 1, 1, 1]  #: Paddings; Length should be 2*num_layers
    final_conv_1_p = 1
    final_conv_2_p = 0

    def __init__(self, device, **model_kwargs):
        super(EF_ConvLSTM, self).__init__(device, **model_kwargs)

        if len(self.enc_names) != self.num_layers or len(self.dec_names) != self.num_layers:
            raise AttributeError(f"Speficied {self.num_layers} layers,"
                                 f"but provided architecture hyperparameters don't match that number")
        for param in ["c", "k", "s", "p"]:
            enc_param = getattr(self, f"enc_{param}")
            dec_param = getattr(self, f"dec_{param}")
            if len(enc_param) != 2 * self.num_layers or len(dec_param) != 2 * self.num_layers:
                raise AttributeError(f"Speficied {self.num_layers} layers,"
                                     f"but provided architecture hyperparameters don't match that number")

        layer_in_c = self.img_c

        # build enc layers and encoder
        enc_convs, enc_cells = [], []
        for n in range(self.num_layers):
            i = 2*n
            layer_out_c = self.enc_c[i+1]
            enc_convs.append(OrderedDict(
                {self.enc_names[n]: [layer_in_c, self.enc_c[i], self.enc_k[i], self.enc_s[i], self.enc_p[i]]}
            ))
            enc_cells.append(ConvLSTM(device=self.device, in_c=self.enc_c[i], enc_c=layer_out_c,
                                      state_h=self.enc_rnn_state_h[n], state_w=self.enc_rnn_state_w[n],
                                      kernel_size=self.enc_k[i+1], stride=self.enc_s[i+1], padding=self.enc_p[i+1]))
            layer_in_c = layer_out_c
        self.encoder = Encoder(enc_convs, enc_cells)

        # build dec layers and decoder, including final convs
        dec_convs, dec_cells = [], []
        for n in range(self.num_layers):
            i = 2*n
            layer_out_c = self.dec_c[i+1]
            dec_cells.append(ConvLSTM(device=self.device, in_c=layer_in_c, enc_c=self.dec_c[i],
                                      state_h=self.dec_rnn_state_h[n], state_w=self.dec_rnn_state_w[n],
                                      kernel_size=self.dec_k[i], stride=self.dec_s[i], padding=self.dec_p[i]))
            dec_conv_dict = {
                self.dec_names[n]: [self.dec_c[i], self.dec_c[i+1], self.dec_k[i+1], self.dec_s[i+1], self.dec_p[i+1]]
            }
            if n == self.num_layers - 1:
                dec_conv_dict[self.final_conv_1_name] = [self.dec_c[i+1], self.final_conv_1_c, self.final_conv_1_k,
                                                         self.final_conv_1_s, self.final_conv_1_p]
                dec_conv_dict[self.final_conv_2_name] = [self.final_conv_1_c, self.img_c, self.final_conv_2_k,
                                                         self.final_conv_2_s, self.final_conv_2_p]
            dec_convs.append(OrderedDict(dec_conv_dict))
            layer_in_c = layer_out_c
        self.forecaster = Forecaster(dec_convs, dec_cells)

        self.NON_CONFIG_VARS.extend(["encoder", "forecaster"])

    def pred_1(self, x, **kwargs):
        return self(x, pred_frames=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_frames: int = 1, **kwargs):
        context_frames = x.shape[1]
        if context_frames < pred_frames:
            pred_1, _ = self(x, pred_frames=context_frames, **kwargs)
            pred_2, _ = self(pred_1, pred_frames=pred_frames-context_frames, **kwargs)
            return torch.cat([pred_1, pred_2], dim=1)
        else:
            x = x.permute((1, 0, 2, 3, 4))  # [t_in, b, c, h, w]
            state = self.encoder(x)
            pred = self.forecaster(state)[:pred_frames]  # [t_pred, b, c, h, w]
            pred = pred.permute((1, 0, 2, 3, 4))  # [b, t_pred, c, h, w]
            return pred, None
