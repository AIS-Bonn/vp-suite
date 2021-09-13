import numpy as np
import torch
import random

from tqdm import tqdm

from metrics.prediction.mse import MSE
from models.prediction.pred_model import VideoPredictionModel

from models.prediction.phydnet.model_blocks import PhyCell, ConvLSTM, EncoderRNN, K2M


class STPhy(VideoPredictionModel):

    # MAGIC NUMBERZ
    enc_channels = 64
    num_layers = 4
    num_hidden = [64, 64, 64, 64]
    decouple_loss_scale = 1.0

    def __init__(self, img_size, img_channels, action_size, device):
        super(STLSTMModel, self).__init__()

        img_height, img_width = img_size

        self.autoencoder = Autoencoder(img_channels, img_size, self.enc_channels, device)
        _, _, self.enc_h, self.enc_w = self.autoencoder.encoded_shape
        self.action_size = action_size
        self.use_actions = self.action_size > 0

        cells = []
        for i in range(self.num_layers):
            in_channel = self.enc_channels + self.action_size if i == 0 else self.num_hidden[i - 1]
            cells.append(STLSTMCell(in_channel, self.num_hidden[i], self.enc_h, self.enc_w,
                                    filter_size=5, stride=1, layer_norm=True))
        self.cell_list = nn.ModuleList(cells)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.enc_channels, kernel_size=1, stride=1,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

        self.device = device
        self.to(self.device)

    def forward(self, x, **kwargs):
        return self.pred_n(x, pred_length=1, **kwargs)

    def pred_n(self, frames, pred_length=1, **kwargs):

        frames = frames.transpose(0, 1)  # [t, b, c, h, w]

        actions = kwargs.get("actions", None)
        if self.use_actions:
            if actions is None or kwargs["actions"].shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
            else:
                actions = F.pad(actions.transpose(0, 1), (0, 0, 0, 0, 1, 0))  # front-pad 1st dim with 0s -> [t, b, a]

        t_in, b, _, _, _ = frames.shape
        T = t_in + pred_length
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([b, self.num_hidden[i], self.enc_h, self.enc_w]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([b, self.num_hidden[0], self.enc_h, self.enc_w]).to(self.device)

        for t in range(T):

            next_cell_input = self.autoencoder.encode(frames[t]) if t < t_in else x_gen
            if self.use_actions:
                inflated_action = actions[t].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.enc_h, self.enc_w)
                next_cell_input = torch.cat([next_cell_input, inflated_action], dim=1)

            for i in range(self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](next_cell_input, h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                next_cell_input = h_t[i]

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(self.autoencoder.decode(x_gen))

            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss_ = torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
                decouple_loss.append(torch.mean(torch.abs(decouple_loss_)))

        predictions = torch.stack(next_frames[t_in:], dim=0).transpose(0, 1)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0)) * self.decouple_loss_scale

        return predictions, {"ST-LSTM decouple loss": decouple_loss}