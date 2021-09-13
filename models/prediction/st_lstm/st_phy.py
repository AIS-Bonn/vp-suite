import numpy as np
import torch
import random

from tqdm import tqdm

from metrics.prediction.mse import MSE
from models.prediction.pred_model import VideoPredictionModel
from models.prediction.st_lstm.model_blocks import STLSTMCell, Autoencoder
from models.prediction.phydnet.model_blocks import PhyCell


class STPhy(VideoPredictionModel):

    # MAGIC NUMBERZ
    enc_channels = 64
    dim_st_hidden = [64, 64, 64, 64]
    n_st_layers = len(dim_st_hidden)
    dim_phy_hidden = [49, 49, 49, 49]
    n_phy_layers = len(dim_phy_hidden)
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

        st_cells = []
        for i in range(self.n_st_layers):
            in_channel = self.enc_channels + self.action_size if i == 0 else self.dim_st_hidden[i - 1]
            st_cells.append(STLSTMCell(in_channel, self.dim_st_hidden[i], self.enc_h, self.enc_w,
                                       filter_size=5, stride=1, layer_norm=True))
        self.st_cell_list = nn.ModuleList(st_cells)

        phy_cells = []
        for i in range(0, self.n_phy_layers):
            cell_list.append(PhyCell_Cell(input_dim=self.enc_channels, action_size=action_size,
                                          F_hidden_dim=self.dim_phy_hidden,
                                          kernel_size=self.phy_kernel_size).to(self.device))
        self.phy_cell_list = nn.ModuleList(phy_cells)

        self.conv_last = nn.Conv2d(in_channels=self.dim_st_hidden[-1] + self.enc_channels,
                                   out_channels=self.enc_channels, kernel_size=(1, 1), stride=(1, 1),
                                   bias=False)
        # shared adapter
        adapter_num_hidden = self.dim_st_hidden[0]
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

        for i in range(self.n_st_layers):
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

            for i in range(self.n_st_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.st_cell_list[i](next_cell_input, h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                next_cell_input = h_t[i]

            x_gen = self.conv_last(h_t[self.n_st_layers - 1])
            next_frames.append(self.autoencoder.decode(x_gen))

            # decoupling loss
            for i in range(0, self.n_st_layers):
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

        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003)
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

            input_length = input_tensor.size(1)
            target_length = target_tensor.size(1)
            loss = 0
            ac_index = 0
            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, output_image, _, _ = self.encoder(input_tensor[:, ei, :, :, :], actions[:, ac_index], (ei == 0))
                loss += self.criterion(output_image, input_tensor[:, ei + 1, :, :, :])
                ac_index += 1

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = self.encoder(decoder_input, actions[:, ac_index])
                target = target_tensor[:, di, :, :, :]
                loss += self.criterion(output_image, target)
                ac_index += 1
                if use_teacher_forcing:
                    decoder_input = target  # Teacher forcing
                else:
                    decoder_input = output_image

            # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
            k2m = K2M([7, 7]).to(self.device)
            for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
                filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
                m = k2m(filters.double())
                m = m.float()
                loss += self.criterion(m, self.constraints)  # constraints is a precomputed matrix

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            optimizer.step()

            loop.set_postfix(loss=loss.item())