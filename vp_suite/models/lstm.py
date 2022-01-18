import torch
from torch import nn as nn
from torchvision import transforms as TF

from vp_suite.models._base_model import VideoPredictionModel


class LSTM(VideoPredictionModel):

    # model-specific constants
    NAME = "NonConvLSTM"
    CAN_HANDLE_ACTIONS = True

    # model hyperparameters
    bottleneck_dim = 1024
    lstm_hidden_dim = 1024
    lstm_kernel_size = (5, 5)
    lstm_num_layers = 3

    def __init__(self, device, **model_args):
        super(LSTM, self).__init__(device, **model_args)

        self.act_fn = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = nn.Conv2d(self.img_c, 64, kernel_size=7, stride=2, padding=3)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="replicate")
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode="replicate")
        self.encoder = nn.Sequential(
            self.enc1, self.act_fn,
            self.enc2, self.act_fn,
            self.enc3, self.act_fn,
        )
        zeros = torch.zeros(1, self.img_c, self.img_h, self.img_w)
        encoded_zeros = self.encoder(zeros)
        self.encoded_shape = encoded_zeros.shape[1:]
        self.encoded_numel = encoded_zeros.numel() // encoded_zeros.shape[0]
        self.to_linear = nn.Linear(self.encoded_numel, self.bottleneck_dim)
        if self.action_conditional:
            inflated_action_size = self.bottleneck_dim // 10
            self.bottleneck_dim += inflated_action_size
            self.action_inflate = nn.Linear(self.action_size, inflated_action_size)
        self.rnn_layers = [
            nn.LSTMCell(input_size=self.bottleneck_dim, hidden_size=self.lstm_hidden_dim, device=self.device)
            for _ in range(self.lstm_num_layers)
        ]
        self.from_linear = nn.Linear(self.lstm_hidden_dim, self.encoded_numel)
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, self.img_c, kernel_size=7, stride=2, padding=3)
        self.decoder = nn.Sequential(
            self.dec1, self.act_fn,
            self.dec2, self.act_fn,
            self.dec3, TF.Resize((self.img_h, self.img_w))
        )

    def _config(self):
        return {
            "bottleneck_dim": self.bottleneck_dim,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_kernel_size": self.lstm_kernel_size,
            "lstm_num_layers": self.lstm_num_layers
        }

    def encode(self, x):
        return self.to_linear(self.encoder(x).flatten(1, -1))  # respect batch size

    def decode(self, x):
        return self.decoder(self.from_linear(x).reshape(x.shape[0], *self.encoded_shape))  # respect batch size

    def pred_1(self, x, **kwargs):
        return self(x, pred_length=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_length=1, **kwargs):

        # frames
        x = x.transpose(0, 1)  # imgs: [t, b, c, h, w]
        T_in, b, c, h, w = x.shape
        assert self.img_shape == (h, w, c),\
            f"input image does not match specified size " \
            f"(input image shape: {x.shape[2:]}, specified: {self.img_shape})"
        encoded_frames = [self.encode(frame) for frame in list(x)]

        # actions
        actions = kwargs.get("actions", None)
        if self.action_conditional:
            if actions is None or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
        if type(actions) == torch.Tensor:
            actions = actions.transpose(0, 1)  # [T_in+pred, b, ...]

        hiddens = [(torch.zeros(b, self.lstm_hidden_dim, device=self.device),
                    torch.zeros(b, self.lstm_hidden_dim, device=self.device)) for _ in self.rnn_layers]

        for t, encoded in enumerate(encoded_frames):
            if self.action_conditional:
                inflated_action = self.action_inflate(actions[t].flatten(1, -1))
                encoded = torch.cat([encoded, inflated_action], dim=-1)
            for (lstm_cell, hidden) in zip(self.rnn_layers, hiddens):
                hidden = lstm_cell(encoded, hidden)

        # pred 1
        output = hiddens[-1][0]
        preds = [self.decode(output)]

        # preds 2, 3, ...
        for t in range(pred_length - 1):
            encoded = self.encode(preds[-1])
            if self.action_conditional:
                inflated_action = self.action_inflate(actions[t - T_in].flatten(1, -1))
                encoded = torch.cat([encoded, inflated_action], dim=-1)
            for (lstm_cell, hidden) in zip(self.rnn_layers, hiddens):
                hidden = lstm_cell(encoded, hidden)
            output = hiddens[-1][0]
            preds.append(self.decode(output))

        # prepare for return
        preds = torch.stack(preds, dim=1)  # output is [b, t, c, h, w] again
        return preds, None
