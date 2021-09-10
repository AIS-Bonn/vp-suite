import torch
from torch import nn as nn
from torchvision.transforms import functional as TF

from models.model_blocks import ConvLSTMCell
from models.prediction.pred_model import VideoPredictionModel


class LSTMModel(VideoPredictionModel):

    def __init__(self, in_channels=3, out_channels=3):
        super(LSTMModel, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)  # final_conv
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