import random

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.image_distance import MSE
from models.vid_pred.pred_model import VideoPredictionModel
from models.vid_pred.st_lstm.model_blocks import STLSTMCell, ActionConditionalSTLSTMCell, Autoencoder
from models.vid_pred.phydnet.model_blocks import PhyCell_Cell, K2M


class STPhy(VideoPredictionModel):
    def __init__(self, img_size, img_channels, enc_channels, phy_channels, num_layers, action_size,
                 inflated_action_dim, phy_kernel_size, decouple_loss_scale, reconstruction_loss_scale,
                 moment_loss_scale, device):
        super(STPhy, self).__init__()

        img_height, img_width = img_size
        self.enc_channels = enc_channels
        self.phy_channels = phy_channels
        self.n_layers = num_layers
        self.dim_st_hidden = [self.enc_channels] * self.n_layers
        self.dim_phy_hidden = [self.phy_channels] * self.n_layers
        self.action_size = action_size
        self.inflated_action_dim = inflated_action_dim
        self.phy_kernel_size = phy_kernel_size
        self.decouple_loss_scale = decouple_loss_scale
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self.moment_loss_scale = moment_loss_scale
        self.device = device

        self.autoencoder = Autoencoder(img_channels, img_size, self.enc_channels, device)
        _, _, self.enc_h, self.enc_w = self.autoencoder.encoded_shape
        self.action_size = action_size
        self.use_actions = self.action_size > 0
        self.recurrent_cell = STLSTMCell

        if self.use_actions:
            self.recurrent_cell = ActionConditionalSTLSTMCell
            self.action_inflate = nn.Linear(in_features=action_size,
                                            out_features=self.inflated_action_dim * self.enc_h * self.enc_w,
                                            bias=False)
            self.action_conv_h = nn.Conv2d(in_channels=self.inflated_action_dim, out_channels=self.enc_channels,
                                           kernel_size=(5, 1), padding=(2, 0), bias=False)
            self.action_conv_w = nn.Conv2d(in_channels=self.inflated_action_dim, out_channels=self.enc_channels,
                                           kernel_size=(1, 5), padding=(0, 2), bias=False)

        st_cells, phy_cells, hidden_convs = [], [], []
        for i in range(self.n_layers):
            st_in_channel = self.dim_st_hidden[0] if i == 0 else self.dim_st_hidden[i - 1]
            st_cells.append(self.recurrent_cell(st_in_channel, self.dim_st_hidden[i], self.enc_h, self.enc_w,
                                       filter_size=5, stride=1, layer_norm=True))
            phy_cells.append(PhyCell_Cell(input_dim=self.enc_channels, action_size=action_size,
                                          F_hidden_dim=self.dim_phy_hidden[i],
                                          kernel_size=self.phy_kernel_size).to(device))
            hc_bias = i < self.n_layers-1
            hidden_convs.append(nn.Conv2d(in_channels=self.enc_channels+self.dim_st_hidden[i],
                                          out_channels=self.enc_channels, kernel_size=(1, 1), bias=hc_bias))
        self.st_cell_list = nn.ModuleList(st_cells)
        self.phy_cell_list = nn.ModuleList(phy_cells)
        self.hidden_conv_list = nn.ModuleList(hidden_convs)

        # shared adapter
        adapter_dim_hidden = self.dim_st_hidden[0]
        self.adapter = nn.Conv2d(adapter_dim_hidden, adapter_dim_hidden, 1, stride=1, padding=0, bias=False)

        self.constraints = torch.zeros((49, 7, 7)).to(device)
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                self.constraints[ind, i, j] = 1
                ind += 1
        self.criterion = MSE()

        self.to(self.device)

    def forward(self, x, **kwargs):
        return self.pred_n(x, pred_length=1, **kwargs)


    def pred_n(self, input, pred_length=1, **kwargs):

        frames = input.transpose(0, 1)  # [t, b, c, h, w]
        input_length, batch_size, _, _, _ = frames.shape
        T = input_length + pred_length
        out_frames = []

        empty_actions = torch.zeros(batch_size, T, device=self.device)
        actions = kwargs.get("actions", empty_actions)
        if self.use_actions:
            if actions.equal(empty_actions) or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
        actions = actions.transpose(0, 1)

        teacher_forcing_ratio = kwargs.get("teacher_forcing_ratio", 0)  # is non-zero during training only
        targets = kwargs.get("target_frames", None)
        if targets is not None:
            targets = targets.transpose(0, 1)

        # init ST and Phy
        st_h_t = []
        st_c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []
        phy_h_t = []

        for i in range(self.n_layers):
            zeros = torch.zeros([batch_size, self.dim_st_hidden[i], self.enc_h, self.enc_w]).to(self.device)
            st_h_t.append(zeros)
            st_c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
            phy_h_t.append(torch.zeros(batch_size, self.enc_channels, self.enc_h, self.enc_w).to(self.device))

        st_memory = torch.zeros([batch_size, self.dim_st_hidden[0], self.enc_h, self.enc_w]).to(self.device)

        # fwd prop through time
        for t in range(T - 1):

            # get input, depending on training/inference stage
            if t < input_length:
                next_cell_input = self.autoencoder.encode(frames[t])
            elif targets is not None and random.random() < teacher_forcing_ratio:
                # During training, with a certain probability, a GT input is chosen instead of the last generated one
                next_cell_input = self.autoencoder.encode(targets[t - input_length])
            else:
                next_cell_input = x_gen

            for i, (st_cell, phy_cell) in enumerate(zip(self.st_cell_list, self.phy_cell_list)):

                # phy
                phy_h_t[i] = phy_cell(next_cell_input, actions[t], phy_h_t[i])

                # st
                if self.use_actions:
                    ac = self.action_inflate(actions[t]).view(-1, self.inflated_action_dim, self.enc_h, self.enc_w)
                    inflated_action = self.action_conv_h(ac) + self.action_conv_w(ac)
                    st_h_t[i], st_c_t[i], st_memory, delta_c, delta_m \
                        = st_cell(next_cell_input, st_h_t[i], st_c_t[i], st_memory, inflated_action)
                else:
                    st_h_t[i], st_c_t[i], st_memory, delta_c, delta_m \
                        = st_cell(next_cell_input, st_h_t[i], st_c_t[i], st_memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

                # merge
                next_cell_input = self.hidden_conv_list[i](torch.cat([st_h_t[i], phy_h_t[i]], dim=-3))

            x_gen = next_cell_input
            out_frame = self.autoencoder.decode(x_gen)
            out_frames.append(out_frame)

            # st decoupling loss
            for i in range(0, self.n_layers):
                decouple_loss_ = torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
            decouple_loss.append(torch.mean(torch.abs(decouple_loss_)))

        out_frames = torch.stack(out_frames, dim=1)
        reconstructions, predictions = out_frames.split([input_length-1, pred_length], dim=1)

        losses = {
            "reconstruction": self.criterion(reconstructions, input[:, 1:]),
            "decouple": torch.mean(torch.stack(decouple_loss, dim=0)),
            "moment": self.get_moment_loss()
        }

        return predictions, losses


    def train_iter(self, cfg, data_loader, optimizer, loss_provider, epoch):

        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.01)
        loop = tqdm(data_loader)
        for batch_idx, data in enumerate(loop):

            # fwd
            img_data = data[cfg.pred_mode].to(self.device) # [b, T, c, h, w], with T = vid_total_length
            input_frames = img_data[:, :cfg.vid_input_length]
            target_frames = img_data[:, cfg.vid_input_length:cfg.vid_total_length]
            actions = data["actions"].to(self.device)

            predictions, model_losses \
                = self.pred_n(input_frames, pred_length=cfg.vid_pred_length, actions=actions,
                              teacher_forcing_ratio=teacher_forcing_ratio, target_frames=target_frames)

            # loss
            _, loss = loss_provider.get_losses(predictions, target_frames)
            loss += model_losses["reconstruction"] * self.reconstruction_loss_scale
            loss += model_losses["decouple"] * self.decouple_loss_scale
            loss += model_losses["moment"] * self.moment_loss_scale

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            optimizer.step()

            loop.set_postfix(loss=loss.item())


    def get_moment_loss(self):
        '''
        Moment regularization
        '''
        loss = torch.tensor(0.0, device=self.device)
        k2m = K2M(self.phy_kernel_size).to(self.device)
        for b in range(0, self.phy_cell_list[0].input_dim):
            filters = self.phy_cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
            m = k2m(filters.double())
            m = m.float()
            loss += self.criterion(m, self.constraints)  # constraints is a precomputed matrix
        return loss