import numpy as np
import torch
import random

from tqdm import tqdm

from losses.image_distance import MSE
from models.vid_pred.pred_model import VideoPredictionModel
from models.vid_pred.specific_model_blocks.phydnet import EncoderRNN, K2M


class PhyDNet(VideoPredictionModel):

    def __init__(self, img_size, img_channels, phy_cell_channels, phy_kernel_size, moment_loss_scale, action_size, device):

        super(PhyDNet, self).__init__()

        self.phy_cell_channels = phy_cell_channels
        self.phy_kernel_size = phy_kernel_size
        self.moment_loss_scale = moment_loss_scale

        self.encoder = EncoderRNN(img_size, img_channels, self.phy_cell_channels, self.phy_kernel_size, action_size, device)
        self.constraints = torch.zeros((self.phy_cell_channels, *self.phy_kernel_size)).to(device)
        ind = 0
        for i in range(0, self.phy_kernel_size[0]):
            for j in range(0, self.phy_kernel_size[1]):
                self.constraints[ind, i, j] = 1
                ind += 1

        self.criterion = MSE()
        self.device = device
        self.action_size = action_size
        self.use_actions = self.action_size > 0

    def forward(self, x, **kwargs):
        return self.pred_n(x, pred_length=1, **kwargs)

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
        return out_frames, None  # inference only -> no loss returned


    def train_iter(self, cfg, data_loader, optimizer, loss_provider, epoch):

        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.01)
        loop = tqdm(data_loader)
        for batch_idx, data in enumerate(loop):

            # fwd
            img_data = data[cfg.pred_mode].to(self.device)  # [b, T, c, h, w], with T = vid_total_length
            input_tensor = img_data[:, :cfg.vid_input_length]
            target_tensor = img_data[:, cfg.vid_input_length:cfg.vid_total_length]

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
            k2m = K2M(self.phy_kernel_size).to(self.device)
            for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
                filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
                m = k2m(filters.double())
                m = m.float()
                loss += self.criterion(m, self.constraints) * self.moment_loss_scale  # constraints is a precomputed matrix

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            optimizer.step()

            loop.set_postfix(loss=loss.item())