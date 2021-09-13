import numpy as np
import torch
import random

from tqdm import tqdm

from metrics.prediction.mse import MSE
from models.prediction.pred_model import VideoPredictionModel
from models.prediction.st_lstm.model_blocks import STLSTMCell, ActionConditionalSTLSTMCell, Autoencoder
from models.prediction.phydnet.model_blocks import PhyCell


class STPhy(VideoPredictionModel):

    # MAGIC NUMBERZ
    enc_channels = 64
    dim_st_hidden = [64, 64, 64]
    dim_phy_hidden = [49, 49, 49]
    if len(dim_st_hidden) != len(dim_phy_hidden):
        raise ValueError("Number of layers need to be the same")
    n_layers = len(dim_st_hidden)
    phy_kernel_size = (7, 7)
    decouple_loss_scale = 1.0
    moment_loss_scale = 1.0

    def __init__(self, img_size, img_channels, action_size, device):
        super(STLSTMModel, self).__init__()

        img_height, img_width = img_size

        self.autoencoder = Autoencoder(img_channels, img_size, self.enc_channels, device)
        _, _, self.enc_h, self.enc_w = self.autoencoder.encoded_shape
        self.action_size = action_size
        self.use_actions = self.action_size > 0
        self.recurrent_cell = STLSTMCell

        if self.use_actions:
            self.recurrent_cell = ActionConditionalSTLSTMCell
            self.action_inflate = nn.Linear(in_features=action_size,
                                            out_features=self.action_linear_size * self.enc_h * self.enc_w,
                                            bias=False)
            self.action_conv_h = nn.Conv2d(in_channels=self.action_linear_size, out_channels=self.enc_channels,
                                           kernel_size=(5, 1), padding=(2, 0), bias=False)
            self.action_conv_w = nn.Conv2d(in_channels=self.action_linear_size, out_channels=self.enc_channels,
                                           kernel_size=(5, 1), padding=(2, 0), bias=False)

        st_cells, phy_cells, hidden_convs = [], [], []
        for i in range(self.n_layers):
            st_in_channel = self.enc_channels + self.action_size if i == 0 else self.dim_st_hidden[i - 1]
            st_cells.append(STLSTMCell(st_in_channel, self.dim_st_hidden[i], self.enc_h, self.enc_w,
                                       filter_size=5, stride=1, layer_norm=True))
            phy_cells.append(PhyCell_Cell(input_dim=self.enc_channels, action_size=action_size,
                                          F_hidden_dim=self.dim_phy_hidden[i],
                                          kernel_size=self.phy_kernel_size).to(self.device))
            hc_bias = i < self.n_layers-1
            hidden_convs.append(nn.Conv2d(in_channels=self.enc_channels+self.dim_st_hidden[i],
                                          out_channels=self.enc_channels, kernel_size=(1, 1), bias=hc_bias))
        self.st_cell_list = nn.ModuleList(st_cells)
        self.phy_cell_list = nn.ModuleList(phy_cells)
        self.hidden_conv_list = nn.ModuleList(hidden_convs)

        # shared adapter
        adapter_num_hidden = self.dim_st_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

        self.criterion = MSE()

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

        for i in range(self.n_layers):
            zeros = torch.zeros([b, self.dim_st_hidden[i], self.enc_h, self.enc_w]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([b, self.dim_st_hidden[0], self.enc_h, self.enc_w]).to(self.device)

        for t in range(T):

            next_cell_input = self.autoencoder.encode(frames[t]) if t < t_in else x_gen
            if self.use_actions:
                inflated_action = actions[t].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.enc_h, self.enc_w)
                next_cell_input = torch.cat([next_cell_input, inflated_action], dim=1)

            for i in range(self.n_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.st_cell_list[i](next_cell_input, h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                next_cell_input = h_t[i]

            x_gen = self.conv_last(h_t[self.n_layers - 1])
            next_frames.append(self.autoencoder.decode(x_gen))

            # decoupling loss
            for i in range(0, self.n_layers):
                decouple_loss_ = torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
                decouple_loss.append(torch.mean(torch.abs(decouple_loss_)))

        predictions = torch.stack(next_frames[t_in:], dim=0).transpose(0, 1)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0)) * self.decouple_loss_scale

        return predictions, {"ST-LSTM decouple loss": decouple_loss}


    # For PhyDNet, pred_n() is used for inference only (no training).
    def pred_n(self, frames, pred_length=1, **kwargs):

        # shape: [b, t, c, ...]
        in_length = frames.shape[1]
        out_frames = []

        empty_actions = torch.zeros(frames.shape[0], frames.shape[1], device=self.device)
        actions = kwargs.get("actions", empty_actions)
        if self.use_actions:
            if actions.equal(empty_actions) or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")

        ac_index = 0
        for ei in range(in_length - 1):
            encoder_output, encoder_hidden, output_image, _, _ = self.encoder(frames[:, ei, :, :, :], actions[:, ac_index], (ei == 0))
            ac_index += 1

        decoder_input = frames[:, -1, :, :, :]  # first decoder input = last image of input sequence

        for di in range(pred_length):
            decoder_output, decoder_hidden, output_image, _, _ = self.encoder(decoder_input, actions[:, ac_index])
            out_frames.append(output_image)
            decoder_input = output_image
            ac_index += 1

        out_frames = torch.stack(out_frames, dim=1)
        return out_frames, None


    def train_iter(self, data_loader, video_in_length, video_pred_length, pred_mode, optimizer, losses, epoch):

        teacher_forcing_ratio = 0.8 ** epoch
        loop = tqdm(data_loader)
        for batch_idx, data in enumerate(loop):

            # fwd
            img_data = data[pred_mode].to(self.device)  # [b, T, c, h, w], with T = VIDEO_TOT_LENGTH
            input_tensor, target_tensor = img_data[:, :video_in_length], img_data[:, video_in_length:]

            actions = data["actions"].to(self.device)
            empty_actions = torch.zeros(img_data.shape[0], img_data.shape[1], device=self.device)
            if self.use_actions:
                if actions.equal(empty_actions) or actions.shape[-1] != self.action_size:
                    raise ValueError("Given actions are None or of the wrong size!")

            b, input_length, _, _, _ = input_tensor.shape
            _, target_length, _, _, _ = target_tensor.shape
            T = input_length + target_length
            next_frames = []
            loss = 0

            # init ST and Phy
            st_h_t = []
            st_c_t = []
            delta_c_list = []
            delta_m_list = []
            decouple_loss = []
            phy_h_t = []

            for i in range(self.n_layers):
                zeros = torch.zeros([b, self.dim_st_hidden[i], self.enc_h, self.enc_w]).to(self.device)
                st_h_t.append(zeros)
                st_c_t.append(zeros)
                delta_c_list.append(zeros)
                delta_m_list.append(zeros)
                phy_h_t.append(torch.zeros(b, self.dim_phy_hidden[i], self.enc_h, self.enc_w).to(self.device))

            memory = torch.zeros([b, self.num_hidden[0], self.enc_h, self.enc_w]).to(self.device)

            for t in range(T):

                # get input and target, depending on training stage
                encoding_phase = t < input_length
                if encoding_phase:
                    next_cell_input = self.autoencoder.encode(input_tensor[:, t])
                    target = input_tensor[:, t + 1, :, :, :]
                else:
                    target = target_tensor[:, t-input_length, :, :, :]
                    if t == input_length:
                        next_cell_input = self.autoencoder.encode(input_tensor[:, t])
                    elif random.random() < teacher_forcing_ratio:
                        next_cell_input = self.autoencoder.encode(target[:, t-input_length])
                    else:
                        next_cell_input = x_gen

                if self.use_actions:
                    action = actions[t]
                    ac = self.action_inflate(action).view(-1, self.action_linear_size, self.enc_h, self.enc_w)
                    inflated_action = self.action_conv_h(ac) + self.action_conv_w(ac)

                for i in range(self.n_layers):
                    phy_h_t[i] = cell(next_cell_input, action, phy_h_t[i])
                    for i in range(self.num_layers):
                    if self.use_actions:
                        h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](next_cell_input, h_t[i],
                                                                                     c_t[i], memory, inflated_action)
                    else:
                        h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](next_cell_input, h_t[i],
                                                                                     c_t[i], memory)
                    delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                    delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                    next_cell_input = self.hidden_conv_list[i](torch.cat[st_h_t[i], phy_h_t[i], dim=-3)

                x_gen = next_cell_input
                output_image = self.autoencoder.decode(x_gen)
                next_frames.append(output_image)

                loss += self.criterion(output_image, target)

                # decoupling loss
                for i in range(0, self.num_layers):
                    decouple_loss_ = torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
                decouple_loss.append(torch.mean(torch.abs(decouple_loss_)))

            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
            loss += decouple_loss * self.decouple_loss_scale

            # Moment regularization
            k2m = K2M([7, 7]).to(self.device)
            for b in range(0, self.phy_cell_list[0].input_dim):
                filters = self.phy_cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
                m = k2m(filters.double())
                m = m.float()
                loss += self.criterion(m, self.constraints) * self.moment_loss_scale  # constraints is a precomputed matrix

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            optimizer.step()

            loop.set_postfix(loss=loss.item())