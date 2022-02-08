import math
import torch
import torch.nn as nn
from vp_suite.models.model_blocks.predrnn import SpatioTemporalLSTMCell as STCell,\
    ActionConditionalSpatioTemporalLSTMCell as ACSTCell
from vp_suite.base.base_model import VideoPredictionModel
import torch.nn.functional as F
from tqdm import tqdm


class PredRNN_V2(VideoPredictionModel):
    r"""
    This is a reimplementation of the model "PredRNN-V2", as introduced in "PredRNN: A Recurrent Neural Network for
    Spatiotemporal Predictive Learning" by Wang et al. (https://arxiv.org/pdf/2103.09504.pdf). This implementation
    is based on the official PyTorch implementation on https://github.com/thuml/predrnn-pytorch.

    PredRNN-V aims at learning partly disentangled spatial/temporal dynamics of the input domain and use this to
    render more accurate predicted frames. The "Spatio-Temporal LSTM Cell" forms the heart of this model.

    Note:
        This model will use the whole frame sequence as an input, including the frames to be predicted. If you do not
        have a ground truth prediction for your frame sequence, pad the sequence with t "zero" frames, with t being
        the amount of predicted frames.
    """

    # model-specific constants
    NAME = "PredRNN V2"
    CAN_HANDLE_ACTIONS = False
    NEEDS_COMPLETE_INPUT = True

    patch_size = 4
    num_layers = 3
    num_hidden = [128, 128, 128, 128]
    filter_size = 5
    stride = 1  #: stride for ST Cell
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
    conv_actions_on_input: bool = True
    residual_on_action_conv: bool = True

    def pred_1(self, x, **kwargs):
        return self(x, pred_frames=1, **kwargs)[0].squeeze(dim=1)

    def __init__(self, device, **model_kwargs):
        super(PredRNN_V2, self).__init__(device, **model_kwargs)

        self.patch_c = self.patch_size * self.patch_size * self.img_c  # frame channel
        self.patch_a = self.patch_size * self.patch_size * self.action_size
        self.patch_h = self.rnn_h = self.img_h // self.patch_size  # cell/RNN height
        self.patch_w = self.rnn_w = self.img_w // self.patch_size  # cell/RNN width
        self.conv_actions_on_input = self.conv_actions_on_input and self.action_conditional  # set to false if not A.C.
        self.residual_on_action_conv = self.residual_on_action_conv and self.conv_actions_on_input  # N/A if no a.-conv.

        # in action-conditional mode, setting conv_actions_on_input to True results in a slightly different comp. graph
        if self.conv_actions_on_input:
            self.rnn_h /= 4
            self.rnn_w /= 4
            self.conv_input1 = nn.Conv2d(self.patch_c, self.num_hidden[0] // 2, self.filter_size,
                                         stride=2, padding=self.filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(self.num_hidden[0] // 2, self.num_hidden[0], self.filter_size,
                                         stride=2, padding=self.filter_size // 2, bias=False)
            self.action_conv_input1 = nn.Conv2d(self.patch_a, self.num_hidden[0] // 2, self.filter_size,
                                                stride=2, padding=self.filter_size // 2, bias=False)
            self.action_conv_input2 = nn.Conv2d(self.num_hidden[0] // 2, self.num_hidden[0], self.filter_size,
                                                stride=2, padding=self.filter_size // 2, bias=False)
            self.deconv_output1 = nn.ConvTranspose2d(self.num_hidden[self.num_layers - 1],
                                                     self.num_hidden[self.num_layers - 1] // 2,
                                                     self.filter_size, stride=2, padding=self.filter_size // 2,
                                                     bias=False)
            self.deconv_output2 = nn.ConvTranspose2d(self.num_hidden[self.num_layers - 1] // 2, self.patch_c,
                                                     self.filter_size, stride=2, padding=self.filter_size // 2,
                                                     bias=False)

        # assemble cell list
        cell_list = []
        cell_class = ACSTCell if self.action_conditional else STCell
        for i in range(self.num_layers):
            if i == 0:
                if self.action_conditional and not self.conv_actions_on_input:
                    in_channel = self.patch_c + self.patch_a
                elif self.action_conditional:
                    in_channel = self.num_hidden[0]
                else:
                    in_channel = self.patch_c
            else:
                in_channel = self.num_hidden[i - 1]
            cell_list.append(
                cell_class(in_channel, self.num_hidden[i], self.rnn_h, self.rnn_w,
                           self.filter_size, self.stride, self.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # conv_last (non-existent for when conv_actions_on_input is True)
        if self.action_conditional and not self.conv_actions_on_input:
            self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.patch_c + self.patch_a,
                                       kernel_size=1, stride=1, padding=0, bias=False)
        elif not self.action_conditional:
            self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.patch_c,
                                       kernel_size=1, stride=1, padding=0, bias=False)

        # adapter
        adap_num_hidden = self.num_hidden[self.num_layers - 1] if self.action_conditional else self.num_hidden[0]
        self.adapter = nn.Conv2d(adap_num_hidden, adap_num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # training variables
        self.training_iteration = 1
        self.sampling_eta = 1.0
        self.NON_CONFIG_VARS.extend(["training_iteration, sampling_eta"])

    def forward(self, x, pred_frames: int = 1, **kwargs):

        # prep variables
        b, total_frames, _, img_h, img_w = x.shape  # NOTE: x NEEDS TO HAVE 'TOTAL_FRAMES' FRAMES!
        context_frames = total_frames - pred_frames
        if context_frames < 1:
            raise ValueError("Model {self.NAME} needs input sequences that also include the target frames!")
        train = kwargs.get("train", False)

        # prep inputs
        empty_actions = torch.zeros(b, total_frames, device=self.device)
        if self.action_conditional:
            actions = kwargs.get("actions", empty_actions)
            if actions.equal(empty_actions) or actions.shape[-1] != self.action_size:
                raise ValueError("Given actions are None or of the wrong size!")
            actions = actions[..., None, None].expand(-1, -1, -1, img_h, img_w)  # [b, t, a, h, w]
            xa = torch.cat([x, actions], dim=2)  # [b, t, c+a, h, w]
            xa_patch = self._reshape_patch(xa)  # [b, t, cpp + app, h_, w_]
            x_patch, a_patch = torch.split(xa_patch, self.patch_c, dim=2)
        else:
            x_patch = self._reshape_patch(x)  # [b, t, cpp, h_, w_]
            a_patch = None

        # cell iteration preparation
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []
        for i in range(self.num_layers):
            zeros = torch.zeros([b, self.num_hidden[i], self.patch_h, self.patch_w], device=self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([b, self.num_hidden[0], self.patch_h, self.patch_w], device=self.device)
        mask_true = self._scheduled_sampling(b, context_frames, pred_frames, train)
        x_gen = None

        # actual fwd pass
        for t in range(total_frames - 1):
            if self.reverse_scheduled_sampling:
                # reverse schedule sampling
                if t == 0:
                    net = x_patch[:, t]
                else:
                    net = mask_true[:, t - 1] * x_patch[:, t] \
                          + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < context_frames:
                    net = x_patch[:, t]
                else:
                    net = mask_true[:, t - context_frames] * x_patch[:, t] \
                          + (1 - mask_true[:, t - context_frames]) * x_gen

            if self.conv_actions_on_input:
                net_shape1 = net.shape
                net = self.conv_input1(net)
                if self.residual_on_action_conv:
                    input_net1 = net
                net_shape2 = net.shape
                net = self.conv_input2(net)
                if self.residual_on_action_conv:
                    input_net2 = net
                a_patch = self.action_conv_input1(a_patch)
                a_patch = self.action_conv_input2(a_patch)

            if self.action_conditional:
                h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, a_patch)
            else:
                h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                if self.action_conditional:
                    h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, a_patch)
                else:
                    h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

            if self.conv_actions_on_input:
                if self.residual_on_action_conv:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)  # [b, cpp, h_, w_]
            elif self.action_conditional:  # NOT EQUIVALENT TO ORIGINAL: discard predicted actions
                x_gen = self.conv_last(h_t[self.num_layers - 1])[:, :self.patch_c]  # [b, cpp, h_, w_]
            else:
                x_gen = self.conv_last(h_t[self.num_layers - 1])  # [b, cpp, h_, w_]
            next_frames.append(x_gen)

        # finalize
        predictions_patch = torch.stack(next_frames, dim=1)  # [b, t', cpp, h_, w_]
        predictions = self._reshape_patch_back(predictions_patch)[:, -pred_frames:]  # [b, t_pred, c, h, w]
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        return predictions, {"ST-LSTM decouple loss": decouple_loss}

    def _reshape_patch(self, x):
        b, t, c, h, w = x.shape
        expected_img_c = self.img_c + self.action_size if self.action_conditional else self.img_c
        expected_input_shape = (expected_img_c, self.img_h, self.img_w)
        if expected_input_shape != (c, h, w):
            raise ValueError(f"shape mismatch: expected {expected_input_shape}, got {(c, h, w)}")
        x = x.view(b, t, c, self.patch_h, self.patch_size, self.patch_w, self.patch_size)
        x = x.permute((0, 1, 4, 6, 2, 3, 5)).contiguous()  # [b, t, p, p, c, h_, w_], 'channels' order: (p_h, p_w, c)
        x_patch = x.view(b, t, -1, self.patch_h, self.patch_w)  # infer channel dim automatically since there might be actions included
        return x_patch

    def _reshape_patch_back(self, x_patch):
        b, t, cpp = x_patch.shape[:3]
        c = cpp // (self.patch_size * self.patch_size)
        h = self.patch_h * self.patch_size
        w = self.patch_w * self.patch_size
        x_patch = x_patch.reshape(b, t, self.patch_size, self.patch_size, c, self.patch_h, self.patch_w)  # [b, t, p, p, c, h_, w_]
        x_patch = x_patch.permute((0, 1, 4, 5, 2, 6, 3))  # [b, t, c, h_, p, w_, p]
        x = x_patch.reshape(b, t, c, h, w)
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

        r_random_flip = torch.rand(batch_size, context_frames-1, device=self.device)
        random_flip = torch.rand(batch_size, pred_frames - 1, device=self.device)

        r_real_input_flag_ = torch.zeros(batch_size, context_frames-1, self.patch_c, self.patch_h, self.patch_w,
                                         device=self.device)
        r_real_input_flag_[(r_random_flip < r_eta)] = 1

        real_input_flag_ = torch.zeros(batch_size, pred_frames-1, self.patch_c, self.patch_h, self.patch_w,
                                       device=self.device)
        real_input_flag_[(random_flip < eta)] = 1

        real_input_flag = torch.cat([r_real_input_flag_, real_input_flag_], dim=1)
        return real_input_flag

    def _std_schedule_sampling(self, batch_size: int, context_frames: int, pred_frames: int):
        pred_frames_m1 = pred_frames - 1

        if not self.scheduled_sampling:
            return 0.0, torch.zeros(batch_size, pred_frames_m1, self.patch_c, self.patch_h, self.patch_w,
                                    device=self.device)

        if self.training_iteration < self.sampling_stop_iter:
            self.sampling_eta -= self.sampling_changing_rate
        else:
            self.sampling_eta = 0.0

        random_flip = torch.rand(batch_size, pred_frames_m1, device=self.device)
        real_input_flag = torch.zeros(batch_size, pred_frames_m1, self.patch_c, self.patch_h, self.patch_w,
                                      device=self.device)
        real_input_flag[(random_flip < self.sampling_eta)] = 1
        return real_input_flag

    def _test_schedule_sampling(self, batch_size: int, context_frames: int, pred_frames: int):
        if self.reverse_scheduled_sampling:
            mask_frames = context_frames + pred_frames - 2
        else:
            mask_frames = pred_frames - 1
        real_input_flag = torch.zeros(batch_size, mask_frames, self.patch_c, self.patch_h, self.patch_w,
                                      device=self.device)
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
            predictions, model_losses = self(input, pred_frames=config["pred_frames"], actions=actions, train=True)

            # loss
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            # reverse
            if self.reverse_input:
                input_rev, targets_rev, actions_rev = self.unpack_data(data, config, reverse=True)
                predictions_rev, model_losses_rev = self(input_rev, pred_frames=config["pred_frames"],
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
