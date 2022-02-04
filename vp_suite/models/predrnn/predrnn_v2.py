import math
import torch
import torch.nn as nn
from vp_suite.models.model_blocks.predrnn import SpatioTemporalLSTMCell as STCell,\
    ActionConditionalSpatioTemporalLSTMCell as ActionConditionalSTCell
from vp_suite.base.base_model import VideoPredictionModel
from vp_suite.base.base_dataset import unpack_data_for_model
from vp_suite.utils.utils import set_from_kwarg
import torch.nn.functional as F
from tqdm import tqdm


class PredRNN_V2(VideoPredictionModel):
    r"""
    Non-action ONLY
    """

    # model-specific constants
    NAME = "PredRNN V2"
    CAN_HANDLE_ACTIONS = False

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
        pass

    def __init__(self, device, **model_kwargs):
        super(PredRNN_V2, self).__init__(device, **model_kwargs)

        set_from_kwarg(self, model_kwargs, "patch_size")

        self.cpp = self.patch_size * self.patch_size * self.img_c  # frame channel
        self.h_ = self.img_h // self.patch_size  # cell height
        self.w_ = self.img_w // self.patch_size  # cell width

        cell_list = []

        for i in range(self.num_layers):
            in_channel = self.cpp if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                STCell(in_channel, self.num_hidden[i], self.h_, self.w_,
                       self.filter_size, self.stride, self.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.cpp,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        # shared adapter
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        self.training_iteration = 1
        self.sampling_eta = 1.0

    def forward(self, x, pred_frames: int = 1, **kwargs):
        mask_true = kwargs.get("mask_true", None)
        if mask_true is None:
            raise ValueError(f"forward method for model {self.NAME} needs kwarg 'mask_true'!")

        b, total_frames = x.shape[:2]  # NOTE: x NEEDS TO HAVE 'TOTAL_FRAMES' FRAMES!
        input_frames = total_frames - pred_frames

        x_patch = self._reshape_patch(x)  # [b, t, cpp, h_, w_]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([b, self.num_hidden[i], self.h_, self.w_]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([b, self.num_hidden[0], self.h_, self.w_]).to(self.device)
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
                if t < input_frames:
                    net = x_patch[:, t]
                else:
                    net = mask_true[:, t - input_frames] * x_patch[:, t] + (1 - mask_true[:, t - input_frames]) * x_gen

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
        a = torch.reshape(x, [b, t, c, self.h_, self.patch_size, self.w_, self.patch_size])
        b = torch.permute(a, (0, 1, 2, 4, 6, 3, 5))  # [b, t, c, p, p, h_, w_]
        x_patch = torch.reshape(b, [b, t, self.cpp, self.h_, self.w_])
        return x_patch

    def _reshape_patch_back(self, x_patch):
        b, t = x_patch.shape[:2]
        c = self.cpp // (self.patch_size * self.patch_size)
        h = self.h_ * self.patch_size
        w = self.w_ * self.patch_size
        a = torch.reshape(x_patch, [b, t, c, self.patch_size, self.patch_size, self.h_, self.w_])
        b = torch.permute(a, [0, 1, 2, 5, 3, 6, 4])  # [b, t, c, h_, p, w_, p]
        x = torch.reshape(b, [b, t, c, h, w])
        return x

    def _reserve_schedule_sampling_exp(self, config):
        b = config["batch_size"]
        context_frames_m1 = config["context_frames"] - 1
        pred_frames_m1 = config["pred_frames"] - 1
        h_ = self.h_
        w_ = self.w_
        cpp = self.cpp

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

        r_random_flip = torch.rand(b, context_frames_m1)
        random_flip = torch.rand(b, pred_frames_m1)

        r_real_input_flag_ = torch.zeros(b, context_frames_m1, cpp, h_, w_)
        r_real_input_flag_[(r_random_flip < r_eta)] = 1

        real_input_flag_ = torch.zeros(b, pred_frames_m1, cpp, h_, w_)
        real_input_flag_[(random_flip < eta)] = 1

        real_input_flag = torch.cat([r_real_input_flag_, real_input_flag_], dim=1)
        return real_input_flag

    def _schedule_sampling(self, config: dict):
        b = config["batch_size"]
        pred_frames_m1 = config["pred_frames"] - 1
        h_ = self.img_h // self.patch_size
        w_ = self.img_w // self.patch_size
        cpp = self.img_c * self.patch_size * self.patch_size

        if not self.scheduled_sampling:
            return 0.0, torch.zeros(b, pred_frames_m1, cpp, h_, w_)

        if self.training_iteration < self.sampling_stop_iter:
            self.sampling_eta -= self.sampling_changing_rate
        else:
            self.sampling_eta = 0.0

        random_flip = torch.rand(b, pred_frames_m1)
        real_input_flag = torch.zeros(b, pred_frames_m1, cpp, h_, w_)
        real_input_flag[(random_flip < self.sampling_eta)] = 1
        return real_input_flag

    def train_iter(self, config, loader, optimizer, loss_provider, epoch):
        loop = tqdm(loader)
        for data in loop:
            input = data["frames"].to(config["device"])  # [b, T, c, h, w], with T = total_frames
            targets = input[:, config["context_frames"]: config["context_frames"] + config["pred_frames"]]

            if self.reverse_scheduled_sampling:
                real_input_flag = self._reserve_schedule_sampling_exp(config)
            else:
                real_input_flag = self._schedule_sampling(config)

            predictions, model_losses = self(input, mask_true=real_input_flag)
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            if self.reverse_input:
                input_rev = torch.flip(input.detach().clone(), dims=[1])
                targets_rev = input_rev[:, config["context_frames"]: config["context_frames"] + config["pred_frames"]]
                predictions_rev, model_losses_rev = self(input_rev, mask_true=real_input_flag)
                _, total_loss_rev = loss_provider.get_losses(predictions_rev, targets_rev)
                if model_losses_rev is not None:
                    for value in model_losses_rev.values():
                        total_loss_rev += value

                total_loss = (total_loss + total_loss_rev) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # bookkeeping
            self.training_iteration += 1
            loop.set_postfix(loss=total_loss.item())


    def eval_iter(self, config: dict, loader: DataLoader, loss_provider: PredictionLossProvider):
        r"""
        Default training iteration: Loops through the whole data loader once and, for every datapoint, executes
        forward pass, and loss calculation. Then, aggregates all loss values to assess the prediction quality.

        Args:
            config (dict): The configuration dict of the current validation run (combines model, dataset and run config)
            loader (DataLoader): Validation data is sampled from this loader.
            loss_provider (PredictionLossProvider): An instance of the :class:`LossProvider` class for flexible loss calculation.

        Returns: A dictionary containing the averages value for each loss type specified for usage,
        as well as the value for the 'indicator' loss (the loss used for determining overall model improvement).

        """
        self.eval()
        loop = tqdm(loader)
        all_losses = []
        indicator_losses = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loop):
                # fwd
                input, targets, actions = unpack_data_for_model(data, config)
                predictions, model_losses = self(input, pred_length=config["pred_frames"], actions=actions)

                # metrics
                loss_values, _ = loss_provider.get_losses(predictions, targets)
                all_losses.append(loss_values)
                indicator_losses.append(loss_values[config["val_rec_criterion"]])

        indicator_loss = torch.stack(indicator_losses).mean()
        all_losses = {
            k: torch.stack([loss_values[k] for loss_values in all_losses]).mean().item() for k in all_losses[0].keys()
        }
        self.train()

        return all_losses, indicator_loss
