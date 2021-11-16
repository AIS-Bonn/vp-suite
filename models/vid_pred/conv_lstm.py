import torch
from torch import nn as nn
from torchvision.transforms import functional as TF

from models.model_blocks import ConvLSTMCell
from models.vid_pred.st_lstm.model_blocks import Autoencoder
from models.vid_pred.pred_model import VideoPredictionModel


class LSTMModelOld(VideoPredictionModel):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, num_layers=3, batch_first=False, bias=True,):
        super(LSTMModelOld, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), padding=(1, 1),
                      padding_mode="replicate", bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=(1, 1),
                      padding_mode="replicate", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(1, 1),
                      padding_mode="replicate", bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                      padding_mode="replicate", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))  # final_conv
        )



        self.lstm = ConvLSTMCell(in_c=64, in_h=64, kernel_size=(3, 3), bias=True)

    def init_hidden(self, img):
        encoded_img = self.encode(img)  # [b, c, h, w] to [b, hidden, h_small, w_small]
        return self.lstm.init_hidden(encoded_img.shape)

    def forward(self, x, **kwargs):
        return self.pred_n(x, pred_length=1)

    def pred_n(self, x, pred_length=1, **kwargs):

        x = x.transpose(0, 1)  # imgs: [t, b, c, h, w]
        state = self.init_hidden(x[0])

        for cur_x in list(x):
            encoded = self.encode(cur_x)
            state = self.lstm(encoded, state)

        preds = [TF.resize(self.decode(state[0]), size=x.shape[3:])]

        if pred_length > 1:
            for t in range(pred_length-1):
                encoded = self.encode(preds[-1])
                state = self.lstm(encoded, state)
                preds.append(TF.resize(self.decode(state[0]), size=x.shape[3:]))

        preds = torch.stack(preds, dim=0).transpose(0, 1)  # output is [b, t, c, h, w] again
        return preds, None


class LSTMModel(nn.Module):
    '''
    Modified from https://github.com/ndrplz/ConvLSTM_pytorch
    '''

    lstm_channels = 64
    encoding_channels = 16, lstm_channels
    decoding_channels = lstm_channels, 32, 16, 8

    def __init__(self, in_channels, out_channels, img_size, lstm_kernel_size, num_layers,
                 action_size, device):
        super(LSTMModel, self).__init__()

        self._check_kernel_size_consistency(lstm_kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        lstm_kernel_size = self._extend_for_multilayer(lstm_kernel_size, num_layers)
        lstm_hidden_dim = self._extend_for_multilayer(self.lstm_channels, num_layers)
        if not len(lstm_kernel_size) == len(lstm_hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_kernel_size = lstm_kernel_size
        self.lstm_num_layers = num_layers
        self.device = device

        self.autoencoder = Autoencoder(self.in_channels, self.img_size,
                                       self.lstm_channels, self.device)
        _, _, self.enc_h, self.enc_w = self.autoencoder.encoded_shape
        self.action_size = action_size
        self.use_actions = self.action_size > 0

        cell_list = []
        for i in range(0, self.lstm_num_layers):
            cur_input_dim = self.lstm_channels if i == 0 else self.lstm_hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(in_c=cur_input_dim + action_size, in_h=self.lstm_hidden_dim[i],
                                          kernel_size=self.lstm_kernel_size[i], bias=True))

        self.cell_list = nn.ModuleList(cell_list)

        if self.use_actions:
            self.action_inflate = nn.Linear(in_features=self.action_size,
                                            out_features=self.action_size * self.enc_h * self.enc_w)

    def forward(self, x, **kwargs):
        return self.pred_n(x, pred_length=1, **kwargs)

    def pred_n(self, x, pred_length=1, **kwargs):

        # frames
        x = x.transpose(0, 1)  # imgs: [t, b, c, h, w]
        T_in, b, _, h, w = x.shape
        assert self.img_size == (h, w), "input image does not match specified size"
        encoded_frames = [self.autoencoder.encode(frame) for frame in list(x)]

        # actions
        actions = kwargs.get("actions", None)
        if self.use_actions:
            if actions is None or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
            else:
                actions = actions.transpose(0, 1)  # [T_in+pred, b]

        # build up belief over given frames
        hidden_state = self._init_hidden((b, self.lstm_channels, self.enc_h, self.enc_w))
        for t, encoded in enumerate(encoded_frames):
            if self.use_actions:
                inflated_action = self.action_inflate(actions[t])\
                    .view(-1, self.action_size, self.enc_h, self.enc_w)
                encoded = torch.cat([encoded, inflated_action], dim=-2)
            hidden_state = self.lstm(encoded, hidden_state)

        # pred 1
        out_hidden, _ = hidden_state[-1]
        preds = [TF.resize(self.autoencoder.decode(out_hidden), size=[h, w])]

        # preds 2, 3, ...
        for t in range(pred_length - 1):
            encoded = self.autoencoder.encode(preds[-1])
            if self.use_actions:
                inflated_action = self.action_inflate(actions[t - T_in])\
                    .view(-1, self.action_size, self.enc_h, self.enc_w)
                encoded = torch.cat([encoded, inflated_action], dim=-2)
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