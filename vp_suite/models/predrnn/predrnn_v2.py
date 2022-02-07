import math
import torch
import torch.nn as nn
from vp_suite.models.model_blocks.predrnn import SpatioTemporalLSTMCell as STCell,\
    ActionConditionalSpatioTemporalLSTMCell as ActionConditionalSTCell
from vp_suite.base.base_model import VideoPredictionModel
import torch.nn.functional as F
from tqdm import tqdm


class PredRNN_V2(VideoPredictionModel):
    r"""
    Non-action ONLY
    """

    # model-specific constants
    NAME = "PredRNN V2"
    CAN_HANDLE_ACTIONS = False
    NEEDS_COMPLETE_INPUT = True

    patch_size = 4
    num_layers = 3
    num_hidden = [128, 128, 128, 128]
    filter_size = 5
    stride = 1
    layer_norm: bool = False
    reverse_input: bool = True
    reconstruction_loss_scale = 0.1
    decoupling_loss_scale = 100.0
    inflated_action_dim = 3

    scheduled_sampling: bool = True
    sampling_stop_iter: int = 50000
    sampling_changing_rate = 2e-5
    reverse_scheduled_sampling: bool = False
    r_sampling_step_1: int = 25000
    r_sampling_step_2: int = 50000
    r_exp_alpha: int = 5000
    training_iteration: int = None
    sampling_eta: float = None

    def pred_1(self, x, **kwargs):
        return self(x, pred_length=1, **kwargs)

    def __init__(self, device, **model_kwargs):
        super(PredRNN_V2, self).__init__(device, **model_kwargs)

        self.NON_CONFIG_VARS.extend(["training_iteration, sampling_eta"])
        self.patch_c = self.patch_size * self.patch_size * self.img_c  # frame channel
        self.patch_h = self.img_h // self.patch_size  # cell height
        self.patch_w = self.img_w // self.patch_size  # cell width

        cell_list = []

        for i in range(self.num_layers):
            in_channel = self.patch_c if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                STCell(in_channel, self.num_hidden[i], self.patch_h, self.patch_w,
                       self.filter_size, self.stride, self.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.patch_c,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        # shared adapter
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        self.training_iteration = 1
        self.sampling_eta = 1.0

    def forward(self, x, pred_frames: int = 1, **kwargs):
        b, total_frames = x.shape[:2]  # NOTE: x NEEDS TO HAVE 'TOTAL_FRAMES' FRAMES!
        context_frames = total_frames - pred_frames
        train = kwargs.get("train", False)
        mask_true = self._scheduled_sampling(b, context_frames, pred_frames, train)
        x_patch = self._reshape_patch(x)  # [b, t, cpp, h_, w_]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([b, self.num_hidden[i], self.patch_h, self.patch_w]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([b, self.num_hidden[0], self.patch_h, self.patch_w]).to(self.device)
        x_gen = None

        for t in range(total_frames - 1):
            if self.reverse_scheduled_sampling:
                # reverse schedule sampling
                if t == 0:
                    net = x_patch[:, t]
                else:
                    net = mask_true[:, t - 1] * x_patch[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < context_frames:
                    net = x_patch[:, t]
                else:
                    net = mask_true[:, t - context_frames] * x_patch[:, t] + (1 - mask_true[:, t - context_frames]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        predictions_patch = torch.stack(next_frames, dim=1)  # [b, t, c, h, w]
        predictions = self._reshape_patch_back(predictions_patch)
        return predictions, {"ST-LSTM decouple loss": decouple_loss}

    def _reshape_patch(self, x):
        b, t, c, h, w = x.shape
        if self.img_shape != (c, h, w):
            raise ValueError(f"shape mismatch: expected {self.img_shape}, got {(c, h, w)}")
        a = torch.reshape(x, [b, t, c, self.patch_h, self.patch_size, self.patch_w, self.patch_size])
        b = torch.permute(a, (0, 1, 2, 4, 6, 3, 5))  # [b, t, c, p, p, h_, w_]
        x_patch = torch.reshape(b, [b, t, self.patch_c, self.patch_h, self.patch_w])
        return x_patch

    def _reshape_patch_back(self, x_patch):
        b, t = x_patch.shape[:2]
        c = self.patch_c // (self.patch_size * self.patch_size)
        h = self.patch_h * self.patch_size
        w = self.patch_w * self.patch_size
        a = torch.reshape(x_patch, [b, t, c, self.patch_size, self.patch_size, self.patch_h, self.patch_w])
        b = torch.permute(a, [0, 1, 2, 5, 3, 6, 4])  # [b, t, c, h_, p, w_, p]
        x = torch.reshape(b, [b, t, c, h, w])
        return x

    def _reserve_schedule_sampling(self, batch_size: int, context_frames: int, pred_frames: int):
        itr = self.training_iteration
        if itr < self.r_sampling_step_1:
            r_eta = 0.5
        elif itr < self.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-float(itr - self.r_sampling_step_1) / self.r_exp_alpha)
        else:
            r_eta = 1.0

        if itr < self.r_sampling_step_1:
            eta = 0.5
        elif itr < self.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.r_sampling_step_2 - self.r_sampling_step_1)) * (itr - self.r_sampling_step_1)
        else:
            eta = 0.0

        r_random_flip = torch.rand(batch_size, context_frames-1)
        random_flip = torch.rand(batch_size, pred_frames - 1)

        r_real_input_flag_ = torch.zeros(batch_size, context_frames-1, self.patch_c, self.patch_h, self.patch_w)
        r_real_input_flag_[(r_random_flip < r_eta)] = 1

        real_input_flag_ = torch.zeros(batch_size, pred_frames-1, self.patch_c, self.patch_h, self.patch_w)
        real_input_flag_[(random_flip < eta)] = 1

        real_input_flag = torch.cat([r_real_input_flag_, real_input_flag_], dim=1)
        return real_input_flag

    def _std_schedule_sampling(self, batch_size: int, context_frames: int, pred_frames: int):
        pred_frames_m1 = pred_frames - 1

        if not self.scheduled_sampling:
            return 0.0, torch.zeros(batch_size, pred_frames_m1, self.patch_c, self.patch_h, self.patch_w)

        if self.training_iteration < self.sampling_stop_iter:
            self.sampling_eta -= self.sampling_changing_rate
        else:
            self.sampling_eta = 0.0

        random_flip = torch.rand(batch_size, pred_frames_m1)
        real_input_flag = torch.zeros(batch_size, pred_frames_m1, self.patch_c, self.patch_h, self.patch_w)
        real_input_flag[(random_flip < self.sampling_eta)] = 1
        return real_input_flag

    def _test_schedule_sampling(self, batch_size: int, context_frames: int, pred_frames: int):
        if self.reverse_scheduled_sampling:
            mask_frames = context_frames + pred_frames - 2
        else:
            mask_frames = pred_frames - 1
        real_input_flag = torch.zeros(batch_size, mask_frames, self.patch_c, self.patch_h, self.patch_w)
        if self.reverse_scheduled_sampling:
            real_input_flag[:, :context_frames-1] = 1
        return real_input_flag

    def _scheduled_sampling(self, batch_size: int, context_frames: int, pred_frames: int, train: bool):
        if not train:
            return self._test_schedule_sampling(batch_size, context_frames, pred_frames)
        elif self.reverse_scheduled_sampling:
            return self._reserve_schedule_sampling(batch_size, context_frames, pred_frames)
        else:
            return self._std_schedule_sampling(batch_size, context_frames, pred_frames)

    def train_iter(self, config, loader, optimizer, loss_provider, epoch):
        r"""
        PredRNN++'s training iteration utilizes reversed input and keeps track of the number of training iterations
        done so far in order to adjust the sampling schedule.
        Otherwise, the iteration logic is the same as in the default :meth:`train_iter()` function.

        Args:
            config (dict): The configuration dict of the current training run (combines model, dataset and run config)
            loader (DataLoader): Training data is sampled from this loader.
            optimizer (Optimizer): The optimizer to use for weight update calculations.
            loss_provider (PredictionLossProvider): An instance of the :class:`LossProvider` class for flexible loss calculation.
            epoch (int): The current epoch.
        """
        loop = tqdm(loader)
        for data in loop:

            # fwd
            input, targets, actions = self.unpack_data(data, config)
            predictions, model_losses = self(input, pred_length=config["pred_frames"], actions=actions, train=True)

            # loss
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            # reverse
            if self.reverse_input:
                input_rev, targets_rev, actions_rev = self.unpack_data(data, config, reverse=True)
                predictions_rev, model_losses_rev = self(input_rev, pred_length=config["pred_frames"],
                                                         actions=actions, train=True)

                # reverse_loss
                _, total_loss_rev = loss_provider.get_losses(predictions_rev, targets_rev)
                if model_losses_rev is not None:
                    for value in model_losses_rev.values():
                        total_loss_rev += value
                total_loss = (total_loss + total_loss_rev) / 2

            # bwd
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # bookkeeping
            self.training_iteration += 1
            loop.set_postfix(loss=total_loss.item())
