from collections import OrderedDict

import torch
import torch.nn as nn

from vp_suite.base.base_model import VideoPredictionModel
from vp_suite.models.precipitation_nowcasting.ef_blocks import Encoder, Forecaster


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
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
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(input, None)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input


class Encoder_Forecaster(VideoPredictionModel):
    r"""
    This is a reimplementation of the Encoder-Forecaster structure, as introduced in
    "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model" by Shi et al.
    (https://arxiv.org/abs/1706.03458). This implementation is based on the PyTorch implementation on
    https://github.com/Hzzone/Precipitation-Nowcasting .

    The Encoder-Forecaster Network stacks multiple convolutional/up-/downsampling and recurrent layers
    that operate on different spatial scales. This model is an abstract one, and the concrete subclasses will fill
    the encoder/forecaster with actual model components.

    Note:
        In its basic version, this model predicts as many frames as it got fed (t_pred == t_in).
        For t_pred < t_in, the same procedure will be used and excess frames will be discarded.
        For t_pred > t_in, multiple consecutive forward passes will be combined that each work on the previous result.
    """
    # model-specific constants
    NAME = "Encoder-Forecaster Structure (Shi et al.)"

    def __init__(self, device, **model_kwargs):
        super(Encoder_Forecaster, self).__init__(device, **model_kwargs)

        per_layer_params = [(k, v) for (k, v) in vars(self) if k.startswith("enc_") or k.startswith("dec_")]
        for param, val in per_layer_params:
            ok = True
            if param in ["enc_c", "dec_c"] and len(val) != 2 * self.num_layers:
                ok = False
            elif len(val) != self.num_layers:
                ok = False
            if not ok:
                raise AttributeError(f"Speficied {self.num_layers} layers, "
                                     f"but len of attribute '{param}' doesn't match that number.")

        enc_convs, enc_rnns, dec_convs, dec_rnns = self._build_encoder_decoder()
        self.encoder = Encoder(enc_convs, enc_rnns)
        self.forecaster = Forecaster(dec_convs, dec_rnns)
        self.NON_CONFIG_VARS.extend(["encoder", "forecaster"])

    def _build_encoder_decoder(self):
        raise NotImplementedError

    def pred_1(self, x, **kwargs):
        return self(x, pred_frames=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_frames: int = 1, **kwargs):
        context_frames = x.shape[1]
        if context_frames < pred_frames:
            pred_1, _ = self(x, pred_frames=context_frames, **kwargs)
            pred_2, _ = self(pred_1, pred_frames=pred_frames - context_frames, **kwargs)
            return torch.cat([pred_1, pred_2], dim=1)
        else:
            x = x.permute((1, 0, 2, 3, 4))  # [t_in, b, c, h, w]
            state = self.encoder(x)
            pred = self.forecaster(state)[:pred_frames]  # [t_pred, b, c, h, w]
            pred = pred.permute((1, 0, 2, 3, 4))  # [b, t_pred, c, h, w]
            return pred, None
