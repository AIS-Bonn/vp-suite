import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from vp_suite.utils.utils import set_from_kwarg, get_public_attrs
from vp_suite.measure.loss_provider import PredictionLossProvider
from vp_suite.base import VPData


class VPModel(nn.Module):
    r"""
    The base class for all video prediction models. Each model ought to provide two forward pass/prediction methods
    (the default :meth:`self.forward()` method and :meth:`pred_1()`, which predicts a single frame) as well as two
    utility methods (:meth:`train_iter()` for a single training epoch on a given dataset loader and, analogously,
    :meth:`eval_iter()` for a single epoch of validation iteration).
    """
    NON_CONFIG_VARS = ["functions", "model_dir", "dump_patches", "training"]  #: Variables that do not get included in the dict returned by :meth:`self.config()` (Constants are not included either).

    # MODEL CONSTANTS
    NAME = None  #: The model's name.
    PAPER_REFERENCE = None  #: The publication where this model was introduced first.
    CODE_REFERENCE = None  #: The code location of the reference implementation.
    MATCHES_REFERENCE: str = None  #: A comment indicating whether the implementation in this package matches the reference.
    REQUIRED_ARGS = ["img_shape", "action_size", "tensor_value_range"]  #: The attributes that the model creator needs to supply when creating the model.
    CAN_HANDLE_ACTIONS = False  #: Whether the model can handle actions or not.
    TRAINABLE = True  #: Whether the model is trainable or not.
    NEEDS_COMPLETE_INPUT = False  #: Whether the input sequences also need to include the to-be-predicted frames.
    MIN_CONTEXT_FRAMES = 1  #: Minimum number of context frames required for the model to work. By default, models will be able to deal with any number of context frames.

    # model hyper-parameters
    model_dir = None  #: The save location of model.
    img_shape = None  # The expected shape of the image inputs.
    action_size = None  #: The expected dimensionality of the action inputs.
    action_conditional = False  #: True if this model is leveraging input actions for the predictions, False otherwise.
    tensor_value_range = None  #: The expected value range of the input tensors.

    def __init__(self, device: str, **model_kwargs):
        r"""
        Initializes the model by first setting all model hyperparameters, attributes and the like.
        Then, the model-specific init will actually create the model from the given hyperparameters

        Args:
            device (str): The device identifier for the module.
            **model_kwargs (Any): Model arguments such as hyperparameters, input shapes etc.
        """
        super(VPModel, self).__init__()

        # set required parameters
        self.device = device
        for required_arg in self.REQUIRED_ARGS:

            # pre-setattr checks
            if required_arg == "tensor_value_range":
                required_val = model_kwargs.get(required_arg, (0, 0))
                if type(required_val) not in [tuple, list] or len(required_val) != 2:
                    raise ValueError("value for argument 'tensor_value_range' needs to be tuple or list with 2 elems")

            # set parameter
            set_from_kwarg(self, model_kwargs, required_arg, required=True)

            # post-setattr logic
            if required_arg == "img_shape":
                self.img_c, self.img_h, self.img_w = self.img_shape

        # set optional parameters
        optional_args = [arg for arg in model_kwargs.keys() if arg not in self.REQUIRED_ARGS]
        for model_arg in optional_args:
            set_from_kwarg(self, model_kwargs, model_arg)

    @property
    def config(self):
        r"""
        Returns: A dictionary containing the complete model configuration, including common attributes
        as well as model-specific attributes.
        """
        attr_dict = get_public_attrs(self, "config", non_config_vars=self.NON_CONFIG_VARS, model_mode=True)
        img_c, img_h, img_w = self.img_shape
        extra_config = {
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "NAME": self.NAME
        }
        return {**attr_dict, **extra_config}

    def unpack_data(self, data: VPData, config: dict, reverse: bool = False, complete: bool = False):
        r"""
        Extracts inputs and targets from a data blob.

        Args:
            data (VPData): The given VPData data blob/dictionary containing frames and actions.
            config (dict): The current run configuration specifying how to extract the data from the given data blob.
            reverse (bool): If specified, reverses the input first
            complete (bool): If specified, input_frames will also contain the to-be-predicted frames (just like with NEEDS_COMPLETE_INPUT)

        Returns: The specified amount of input/target frames as well as the actions. All inputs will come in the
        shape the model expects as input later.
        """
        img_data = data["frames"].to(config["device"])  # [b, T, c, h, w], with T = total_frames
        actions = data["actions"].to(config["device"])  # [b, T-1, a]. Action t happens between frame t and t+1
        if img_data.ndim == 4:  # prepend batch dimension if not given
            img_data = img_data.unsqueeze(0)
            actions = actions.unsqueeze(0)
        if reverse:
            img_data = torch.flip(img_data, dims=[1])
            actions = torch.flip(actions, dims=[1])
        T_in, T_pred = config["context_frames"], config["pred_frames"]
        if self.NEEDS_COMPLETE_INPUT or complete:
            input_frames = img_data[:, :T_in+T_pred]
            target_frames = input_frames[:, T_in:].clone()
        else:
            input_frames, target_frames = torch.split(img_data[:, :T_in+T_pred], [T_in, T_pred], dim=1)
        return input_frames, target_frames, actions

    def pred_1(self, x: torch.Tensor, **kwargs):
        r"""
        Given an input sequence of t frames, predicts one single frame into the future.

        Args:
            x (torch.Tensor): A batch of `b` sequences of `t` input frames as a tensor of shape [b, t, c, h, w].
            **kwargs (Any): Optional input parameters such as actions.

        Returns: A single frame as a tensor of shape [b, c, h, w].
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, pred_frames: int = 1, **kwargs):
        r"""
        Given an input sequence of t frames, predicts `pred_frames` (`p`) frames into the future.

        Args:
            x (torch.Tensor): A batch of `b` sequences of `t` input frames as a tensor of shape [b, t, c, h, w].
            pred_frames (int): The number of frames to predict into the future.
            **kwargs ():

        Returns: A batch of sequences of `p` predicted frames as a tensor of shape [b, p, c, h, w].
        """
        predictions = []
        for i in range(pred_frames):
            pred = self.pred_1(x, **kwargs).unsqueeze(dim=1)
            predictions.append(pred)
            x = torch.cat([x, pred], dim=1)

        pred = torch.cat(predictions, dim=1)
        return pred, None

    def train_iter(self, config: dict, loader: DataLoader, optimizer: Optimizer,
                   loss_provider: PredictionLossProvider, epoch: int):
        r"""
        Default training iteration: Loops through the whole data loader once and, for every batch, executes
        forward pass, loss calculation and backward pass/optimization step.

        Args:
            config (dict): The configuration dict of the current training run (combines model, dataset and run config)
            loader (DataLoader): Training data is sampled from this loader.
            optimizer (Optimizer): The optimizer to use for weight update calculations.
            loss_provider (PredictionLossProvider): An instance of the :class:`LossProvider` class for flexible loss calculation.
            epoch (int): The current epoch.
        """
        loop = tqdm(loader)
        for batch_idx, data in enumerate(loop):
            # fwd
            input, targets, actions = self.unpack_data(data, config)
            predictions, model_losses = self(input, pred_frames=config["pred_frames"], actions=actions)

            # loss
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            # bwd
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # bookkeeping
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
                input, targets, actions = self.unpack_data(data, config)
                predictions, model_losses = self(input, pred_frames=config["pred_frames"], actions=actions)

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
