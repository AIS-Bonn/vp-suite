import random

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vp_suite.base import VPModel
from vp_suite.model_blocks import Autoencoder
from vp_suite.model_blocks.predrnn import SpatioTemporalLSTMCell, ActionConditionalSpatioTemporalLSTMCell
from vp_suite.model_blocks.phydnet import PhyCell_Cell, K2M


class STPhy(VPModel):
    r"""
    This class implements a hybrid model that aims to unify the advantages of the
    PhyDNet (Le Guen and Thome, https://arxiv.org/abs/2003.01460, https://github.com/vincent-leguen/PhyDNet) and the
    PredRNN++ (Wang et al., https://arxiv.org/abs/2103.09504, https://github.com/thuml/predrnn-pytorch) models.
    More specifically, it replaces PhyDNet's regular ConvLSTM cells with the ST Cells from PredRNN++
    and integrates PhyDNet's teacher forcing and PredRNN++'s scheduled sampling techniques into training.
    (TODO adjust model to cohere to this description)
    """
    NAME = "ST-Phy"
    CAN_HANDLE_ACTIONS = True

    num_layers = 3  #: Number of layers (1 PhyCell and 1 ST cell per layer)
    phycell_channels = 49  #: Channel dimensionality for the PhyCells
    phycell_kernel_size = (7, 7)  #: PhyCell kernel size
    st_cell_channels = 64  #: Hidden layer dimensionality for the ST cell layers
    inflated_action_dim = 3  #: Dimensionality of the 'inflated actions' (actions that have been transformed to tensors)

    decoupling_loss_scale = 100.0  #: The scaling factor for the decoupling loss
    moment_loss_scale = 1.0  #: Scaling factor for the moment loss (for PDE-Constrained prediction by the PhyCells)
    teacher_forcing_decay = 0.003  #: Per-Episode decrease of the teacher forcing ratio (Starts out at 1.0)

    def __init__(self, device, **model_kwargs):
        super(STPhy, self).__init__(device, **model_kwargs)

        self.dim_st_hidden = [self.st_cell_channels] * self.num_layers
        self.dim_phy_hidden = [self.phycell_channels] * self.num_layers

        self.autoencoder = Autoencoder(self.img_shape, self.st_cell_channels, self.device)
        _, _, self.enc_h, self.enc_w = self.autoencoder.encoded_shape
        self.recurrent_cell = SpatioTemporalLSTMCell

        if self.action_conditional:
            self.recurrent_cell = ActionConditionalSpatioTemporalLSTMCell
            self.action_inflate = nn.Linear(in_features=self.action_size,
                                            out_features=self.inflated_action_dim * self.enc_h * self.enc_w,
                                            bias=False)
            self.action_conv_h = nn.Conv2d(in_channels=self.inflated_action_dim, out_channels=self.st_cell_channels,
                                           kernel_size=(5, 1), padding=(2, 0), bias=False)
            self.action_conv_w = nn.Conv2d(in_channels=self.inflated_action_dim, out_channels=self.st_cell_channels,
                                           kernel_size=(1, 5), padding=(0, 2), bias=False)

        st_cells, phycells, hidden_convs = [], [], []
        for i in range(self.num_layers):
            cell_in_channel = self.dim_st_hidden[0] if i == 0 else self.dim_st_hidden[i - 1]
            st_cells.append(self.recurrent_cell(cell_in_channel, self.dim_st_hidden[i], self.enc_h, self.enc_w,
                                                filter_size=5, stride=1, layer_norm=True))

            phycells.append(PhyCell_Cell(input_dim=cell_in_channel, action_conditional=self.action_conditional,
                                         action_size=self.action_size, hidden_dim=self.dim_phy_hidden[i],
                                         kernel_size=self.phycell_kernel_size).to(self.device))

            hc_bias = i < self.num_layers - 1
            hidden_convs.append(nn.Conv2d(in_channels=self.st_cell_channels + self.dim_st_hidden[i],
                                          out_channels=self.st_cell_channels, kernel_size=(1, 1), bias=hc_bias))

        self.st_cell_list = nn.ModuleList(st_cells)
        self.phycell_list = nn.ModuleList(phycells)
        self.hidden_conv_list = nn.ModuleList(hidden_convs)

        # shared adapter
        adapter_dim_hidden = self.dim_st_hidden[0]
        self.adapter = nn.Conv2d(adapter_dim_hidden, adapter_dim_hidden, 1, stride=1, padding=0, bias=False)

        self.constraints = torch.zeros((self.phycell_channels, *self.phycell_kernel_size), device=self.device)
        ind = 0
        for i in range(0, self.phycell_kernel_size[0]):
            for j in range(0, self.phycell_kernel_size[1]):
                self.constraints[ind, i, j] = 1
                ind += 1

    def pred_1(self, x, **kwargs):
        return self(x, pred_frames=1, **kwargs)[0].squeeze(dim=1)

    def forward(self, x, pred_frames=1, **kwargs):
        # in training mode (default: False), returned sequence starts with 2nd context frame,
        # and the moment regularization loss is calculated.
        train = kwargs.get("train", False)
        teacher_forcing = kwargs.get("teacher_forcing", False) and train
        batch_size, context_frames = x.shape[:2]
        if train:  # during training, the whole sequence is passed
            context_frames -= pred_frames
        empty_actions = torch.zeros(x.shape[0], context_frames+pred_frames-1, device=self.device)
        actions = kwargs.get("actions", empty_actions)
        if self.action_conditional:
            if actions.equal(empty_actions) or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")

        # init ST and Phy
        phy_h_t = []
        st_h_t = []
        st_c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.dim_st_hidden[i], self.enc_h, self.enc_w]).to(self.device)
            st_h_t.append(zeros)
            st_c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
            phy_h_t.append(torch.zeros(batch_size, self.st_cell_channels, self.enc_h, self.enc_w).to(self.device))

        st_memory = torch.zeros([batch_size, self.dim_st_hidden[0], self.enc_h, self.enc_w]).to(self.device)
        out_frames = []
        x_gen = None

        # fwd prop through time
        for t in range(context_frames + pred_frames - 1):

            # get input, depending on training/inference stage
            if t < context_frames or teacher_forcing:
                next_input = self.autoencoder.encode(x[:, t])
            else:
                next_input = x_gen

            for i, (st_cell, phycell) in enumerate(zip(self.st_cell_list, self.phycell_list)):

                # phy
                phy_h_t[i] = phycell(next_input, actions[:, t], phy_h_t[i])  # returns (hidden, hidden)

                # st
                if self.action_conditional:
                    ac = self.action_inflate(actions[:, t]).view(-1, self.inflated_action_dim, self.enc_h, self.enc_w)
                    inflated_action = self.action_conv_h(ac) + self.action_conv_w(ac)
                    st_h_t[i], st_c_t[i], st_memory, delta_c, delta_m \
                        = st_cell(next_input, st_h_t[i], st_c_t[i], st_memory, inflated_action)
                else:
                    st_h_t[i], st_c_t[i], st_memory, delta_c, delta_m \
                        = st_cell(next_input, st_h_t[i], st_c_t[i], st_memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

                # merge
                x_gen = self.hidden_conv_list[i](torch.cat([st_h_t[i], phy_h_t[i]], dim=-3))

            if train or t >= (context_frames - 1):
                out_frame = self.autoencoder.decode(x_gen)
                out_frames.append(out_frame)

            # Spatio-temporal decoupling loss in STCell during training
            if train:
                for i in range(0, self.num_layers):
                    decouple_loss.append(
                        torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        out_frames = torch.stack(out_frames, dim=1)

        if train:
            # Moment regularization loss during training
            k2m = K2M(self.phycell_kernel_size).to(self.device)
            moment_loss = 0
            for b in range(0, self.phycell_list[0].input_dim):
                filters = self.phycell_list[0].F.conv1.weight[:, b]
                moment = k2m(filters.double()).float()
                moment_loss += torch.mean(self.moment_loss_scale * (moment - self.constraints) ** 2)
            decoupling_loss = torch.mean(torch.stack(decouple_loss, dim=0))
            model_losses = {
                "moment regularization loss": self.moment_loss_scale * moment_loss,
                "memory decoupling loss": self.decoupling_loss_scale * decoupling_loss,
            }
        else:
            model_losses = None
        return out_frames, model_losses

    def train_iter(self, config, data_loader, optimizer, loss_provider, epoch):
        r"""
        ST-Phy's training iteration utilizes a scheduled teacher forcing ratio.
        Otherwise, the iteration logic is the same as in the default :meth:`train_iter()` function.

        Args:
            config (dict): The configuration dict of the current training run (combines model, dataset and run config)
            data_loader (DataLoader): Training data is sampled from this loader.
            optimizer (Optimizer): The optimizer to use for weight update calculations.
            loss_provider (PredictionLossProvider): An instance of the :class:`LossProvider` class for flexible loss calculation.
            epoch (int): The current epoch.
        """
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * self.teacher_forcing_decay)
        loop = tqdm(data_loader)
        for batch_idx, data in enumerate(loop):

            # fwd
            input_frames, _, actions = self.unpack_data(data, config, complete=True)
            teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            predictions, model_losses = self(input_frames, pred_frames=config["pred_frames"],
                                             actions=actions, train=True, teacher_forcing=teacher_forcing)

            # loss
            targets = input_frames[:, 1:]
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_postfix(loss=total_loss.item())
