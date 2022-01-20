import torch
import torch.nn as nn
from tqdm import tqdm

class VideoPredictionModel(nn.Module):
    r"""

    """

    # model-specific constants
    NAME = None  #: The model's name
    REQUIRED_ARGS = ["img_shape", "action_size", "tensor_value_range"]  #: TODO
    CAN_HANDLE_ACTIONS = False  #: Whether the model can handle actions or not
    TRAINABLE = True  #: Whether the model is trainable or not

    # model hyperparameters
    min_context_frames = 1  # : Minimum number of context frames required for the model to work. Be default models
    # will be able to deal with arbitrarily many context frames.
    action_conditional = None  #: TODO
    model_dir = None  #: specifies save location of model
    tensor_value_range = None  #: TODO

    def __init__(self, device="cpu", **model_args):
        r"""

        Args:
            device ():
            **model_args ():
        """
        super(VideoPredictionModel, self).__init__()

        # set required parameters
        self.device = device
        for required_arg in self.REQUIRED_ARGS:
            assert required_arg in model_args.keys(), f"ERROR: model {self.NAME} requires parameter '{required_arg}'"
            setattr(self, required_arg, model_args[required_arg])
            if required_arg == "img_shape":
                self.img_h, self.img_w, self.img_c = self.img_shape
            elif required_arg == "tensor_value_range":
                assert isinstance(self.tensor_value_range, list) or isinstance(self.tensor_value_range, tuple)
                assert len(self.tensor_value_range) == 2

        # set optional parameters
        self.action_conditional = model_args.get("action_conditional", False)
        for model_arg, model_arg_val in model_args.items():
            if model_arg in self.REQUIRED_ARGS:
                continue  # skip required args as they have been set up already
            elif hasattr(self, model_arg):
                setattr(self, model_arg, model_arg_val)
            else:
                print(f"INFO: model_arg '{model_arg}' is not usable for init of model '{self.NAME}' -> skipping")

    @property
    def config(self):
        r"""

        Returns:

        """
        model_config = {
            "model_dir": self.model_dir,
            "min_context_frames": self.min_context_frames,
            "img_shape": self.img_shape,
            "action_size": self.action_size,
            "action_conditional": self.action_conditional,
            "tensor_value_range": self.tensor_value_range
        }
        return {**model_config, **self._config()}

    def _config(self):
        r""" Model-specific config
        """
        return dict()

    def pred_1(self, x, **kwargs):
        r"""Predicts a single frame
        input: T frames: [b, T, c, h, w]
        output: single frame: [b, c, h, w]

        Args:
            x ():
            **kwargs ():

        Returns:

        """
        raise NotImplementedError

    def forward(self, x, pred_length=1, **kwargs):
        r""" Predicts pred_length frames into the future.
        # input: T frames: [b, T, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]

        Args:
            x ():
            pred_length ():
            **kwargs ():

        Returns:

        """
        preds = []
        for i in range(pred_length):
            pred = self.pred_1(x, **kwargs).unsqueeze(dim=1)
            preds.append(pred)
            x = torch.cat([x, pred], dim=1)

        pred = torch.cat(preds, dim=1)
        return pred, None

    def train_iter(self, config, loader, optimizer, loss_provider, epoch):
        r"""Default training iteration

        Args:
            config ():
            loader ():
            optimizer ():
            loss_provider ():
            epoch ():

        Returns:

        """
        loop = tqdm(loader)
        for batch_idx, data in enumerate(loop):

            # input
            img_data = data["frames"].to(config["device"])  # [b, T, c, h, w], with T = total_frames
            input = img_data[:, :config["context_frames"]]
            targets = img_data[:, config["context_frames"]
                                  : config["context_frames"] + config["pred_frames"]]
            actions = data["actions"].to(
                config["device"])  # [b, T-1, a]. Action t corresponds to what happens after frame t

            # fwd
            predictions, model_losses = self(input, pred_length=config["pred_frames"], actions=actions)

            # loss
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            # bwd
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            optimizer.step()

            # bookkeeping
            loop.set_postfix(loss=total_loss.item())
            # loop.set_postfix(mem=torch.cuda.memory_allocated())

    def eval_iter(self, config, loader, loss_provider):
        r"""Default evaluation iteration

        Args:
            config ():
            loader ():
            loss_provider ():

        Returns:

        """
        self.eval()
        loop = tqdm(loader)
        all_losses = []
        indicator_losses = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loop):
                # fwd
                img_data = data["frames"].to(config["device"])  # [b, T, h, w], with T = total_frames
                input = img_data[:, :config["context_frames"]]
                targets = img_data[:, config["context_frames"]
                                      : config["context_frames"] + config["pred_frames"]]
                actions = data["actions"].to(config["device"])

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
