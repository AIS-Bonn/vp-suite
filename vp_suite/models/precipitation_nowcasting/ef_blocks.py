r"""
This module contains the building blocks for the video prediction models implemented in
https://github.com/Hzzone/Precipitation-Nowcasting (Encoder-Forecaster-based prediction models based on
ConvLSTM/TrajGRU).
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from vp_suite.base import VPModel
from vp_suite.utils.models import conv_output_shape, convtransp_output_shape

def _make_layers(block):
    r"""
    Utility method to create the layers for the Encoder and Forecaster components.
    """
    layers = []
    for layer_name, v in block.items():
        if 'identity' in layer_name:
            layer = nn.Identity()
            layers.append((layer_name, layer))
        elif 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    r"""
    The Encoder component of the Encoder-Forecaster architecture wraps a recurrent model block to encode the input
    frames into a state that can then be used for prediction.
    """
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), _make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        b, t, c, h, w = input.shape
        input = torch.reshape(input, (-1, c, h, w))
        input = subnet(input)
        input = torch.reshape(input, (b, t, input.size(1), input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(input, None, seq_len=t)
        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)),
                                                       getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    r"""
    The Decoder component of the Encoder-Forecaster architecture wraps a recurrent model block to decode the input
    state into a specified number of predicted future frames.
    """
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), _make_layers(params))

    def forward_by_stage(self, input, state, pred_frames, subnet, rnn):
        input, state_stage = rnn(input, state, pred_frames)
        b, t, c, h, w = input.shape
        input = torch.reshape(input, (-1, c, h, w))
        input = subnet(input)
        input = torch.reshape(input, (b, t, input.size(1), input.size(2), input.size(3)))
        return input

    def forward(self, hidden_states, pred_frames):
        input = self.forward_by_stage(None, hidden_states[-1], pred_frames, getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], pred_frames, getattr(self, 'stage' + str(i)),
                                          getattr(self, 'rnn' + str(i)))
        return input


class Encoder_Forecaster(VPModel):
    r"""
    This is a reimplementation of the Encoder-Forecaster structure, as introduced in
    "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model" by Shi et al.
    (https://arxiv.org/abs/1706.03458). This implementation is based on the PyTorch implementation on
    https://github.com/Hzzone/Precipitation-Nowcasting .

    The Encoder-Forecaster Network stacks multiple convolutional/up-/downsampling and recurrent layers
    that operate on different spatial scales. This model is an abstract one, and the concrete subclasses will fill
    the encoder/forecaster with actual model components.
    """
    # model-specific constants
    NAME = "Encoder-Forecaster Structure (Shi et al.)"

    def __init__(self, device, **model_kwargs):
        super(Encoder_Forecaster, self).__init__(device, **model_kwargs)

        per_layer_params = [(k, v) for (k, v) in vars(self).items() if k.startswith("enc_") or k.startswith("dec_")]
        for param, val in per_layer_params:
            ok = True
            if param in ["enc_c", "dec_c"] and len(val) != 2 * self.num_layers:
                ok = False
            elif param not in ["enc_c", "dec_c"] and len(val) != self.num_layers:
                ok = False
            if not ok:
                raise AttributeError(f"Speficied {self.num_layers} layers, "
                                     f"but len of attribute '{param}' doesn't match that ({val}).")

        # set rnn state sizes according to calculated conv output size
        next_h, next_w = self.img_h, self.img_w
        enc_rnn_state_h, enc_rnn_state_w = [], []
        for n in range(self.num_layers):
            next_h, next_w = conv_output_shape((next_h, next_w),
                                               self.enc_conv_k[n], self.enc_conv_s[n], self.enc_conv_p[n])
            enc_rnn_state_h.append(next_h)
            enc_rnn_state_w.append(next_w)

        dec_rnn_state_h, dec_rnn_state_w = [next_h], [next_w]
        for n in range(self.num_layers - 1):
            next_h, next_w = convtransp_output_shape((next_h, next_w),
                                                     self.dec_conv_k[n], self.dec_conv_s[n], self.dec_conv_p[n])
            dec_rnn_state_h.append(next_h)
            dec_rnn_state_w.append(next_w)

        final_h, final_w = convtransp_output_shape((next_h, next_w),
                                                   self.dec_conv_k[-1], self.dec_conv_s[-1], self.dec_conv_p[-1])
        if (self.img_h, self.img_w) != (final_h, final_w):
            hidden_sizes = list(zip(enc_rnn_state_h, enc_rnn_state_w)) + list(zip(dec_rnn_state_h, dec_rnn_state_w))
            raise AttributeError(f"Model layer hyperparameters yield wrong output size: "
                                 f"{(final_h, final_w)} (expected: {(self.img_h, self.img_w)}). "
                                 f"All hidden sizes: {hidden_sizes}")

        self.enc_rnn_state_h = enc_rnn_state_h
        self.enc_rnn_state_w = enc_rnn_state_w
        self.dec_rnn_state_h = dec_rnn_state_h
        self.dec_rnn_state_w = dec_rnn_state_w
        enc_convs, enc_rnns, dec_convs, dec_rnns = self._build_encoder_decoder()
        self.encoder = Encoder(enc_convs, enc_rnns).to(self.device)
        self.forecaster = Forecaster(dec_convs, dec_rnns).to(self.device)
        self.NON_CONFIG_VARS.extend(["encoder", "forecaster"])

    def _build_encoder_decoder(self):
        raise NotImplementedError

    def pred_1(self, x, **kwargs):
        return self(x, pred_frames=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_frames: int = 1, **kwargs):
        state = self.encoder(x)
        pred = self.forecaster(state, pred_frames)
        return pred, None
