import torch
from torch import nn as nn
from torchvision.transforms import functional as TF

from vp_suite.models.model_blocks.recurrent import ConvLSTMCell
from vp_suite.models.model_blocks.enc import Autoencoder
from vp_suite.models._base_model import VideoPredictionModel


class ConvLSTM(VideoPredictionModel):
    '''
    Modified from https://github.com/ndrplz/ConvLSTM_pytorch
    '''

    # model-specific constants
    NAME = "ConvLSTM"
    CAN_HANDLE_ACTIONS = True

    # model hyperparameters
    lstm_kernel_size = (3, 3)
    lstm_num_layers = 3
    lstm_channels = 64
    encoding_channels = 16, lstm_channels
    decoding_channels = lstm_channels, 32, 16, 8

    def __init__(self, device, **model_args):
        super(ConvLSTM, self).__init__(device, **model_args)
        self._check_kernel_size_consistency(self.lstm_kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        lstm_kernel_size = self._extend_for_multilayer(self.lstm_kernel_size, self.lstm_num_layers)
        lstm_hidden_dim = self._extend_for_multilayer(self.lstm_channels, self.lstm_num_layers)
        if not len(lstm_kernel_size) == len(lstm_hidden_dim) == self.lstm_num_layers:
            raise ValueError('Inconsistent list length.')

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_kernel_size = lstm_kernel_size

        self.autoencoder = Autoencoder(self.img_shape, self.lstm_channels, self.device)
        _, _, self.enc_h, self.enc_w = self.autoencoder.encoded_shape

        cell_list = []
        cell_ac_size = self.action_size if self.action_conditional else 0
        for i in range(0, self.lstm_num_layers):
            cur_input_dim = self.lstm_channels if i == 0 else self.lstm_hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(in_c=cur_input_dim + cell_ac_size, in_h=self.lstm_hidden_dim[i],
                                          kernel_size=self.lstm_kernel_size[i], bias=True))

        self.cell_list = nn.ModuleList(cell_list)

        if self.action_conditional:
            self.action_inflate = nn.Linear(in_features=self.action_size,
                                            out_features=self.action_size * self.enc_h * self.enc_w)

    def _config(self):
        return {
            "lstm_kernel_size": self.lstm_kernel_size,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_channels": self.lstm_channels,
            "encoding_channels": self.encoding_channels,
            "decoding_channels": self.decoding_channels
        }

    def pred_1(self, x, **kwargs):
        return self(x, pred_length=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_length=1, **kwargs):

        # frames
        x = x.transpose(0, 1)  # imgs: [t, b, c, h, w]
        T_in, b, c, h, w = x.shape
        assert self.img_shape == (h, w, c), "input image does not match specified size"
        encoded_frames = [self.autoencoder.encode(frame) for frame in list(x)]

        # actions
        actions = kwargs.get("actions", None)
        if self.action_conditional:
            if actions is None or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
        if type(actions) == torch.Tensor:
            actions = actions.transpose(0, 1)  # [T_in+pred, b, ...]

        # build up belief over given frames
        hidden_state = self._init_hidden((b, self.lstm_channels, self.enc_h, self.enc_w))
        for t, encoded in enumerate(encoded_frames):
            if self.action_conditional:
                inflated_action = self.action_inflate(actions[t])\
                    .view(-1, self.action_size, self.enc_h, self.enc_w)
                encoded = torch.cat([encoded, inflated_action], dim=-3)
            hidden_state = self.lstm(encoded, hidden_state)

        # pred 1
        out_hidden, _ = hidden_state[-1]
        preds = [TF.resize(self.autoencoder.decode(out_hidden), size=[h, w])]

        # preds 2, 3, ...
        for t in range(pred_length - 1):
            encoded = self.autoencoder.encode(preds[-1])
            if self.action_conditional:
                inflated_action = self.action_inflate(actions[t - T_in])\
                    .view(-1, self.action_size, self.enc_h, self.enc_w)
                encoded = torch.cat([encoded, inflated_action], dim=-3)
            hidden_state = self.lstm(encoded, hidden_state)
            out_hidden, _ = hidden_state[-1]
            preds.append(TF.resize(self.autoencoder.decode(out_hidden), size=[h, w]))

        # prepare for return
        preds = torch.stack(preds, dim=0).transpose(0, 1)  # output is [b, t, c, h, w] again
        return preds, None

    def lstm(self, input, hidden_state):
        new_state_list = []
        for layer_idx in range(self.lstm_num_layers):
            h, c = hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](input_tensor=input, cur_state=[h, c])
            new_state_list.append([h, c])
        return new_state_list

    def _init_hidden(self, image_size):
        init_states = []
        for i in range(self.lstm_num_layers):
            init_states.append(self.cell_list[i].init_hidden(image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param