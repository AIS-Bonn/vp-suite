import torch
import torch.nn as nn

from tqdm import tqdm

class VideoPredictionModel(nn.Module):

    NAME = None
    CAN_HANDLE_ACTIONS = False  # models by default won't be able to handle actions
    TRAINABLE = True  # most implemented models will be trainable

    min_context_frames = 1  # models by default will be able to deal with arbitrarily many context frames
    action_conditional = None
    model_dir = None  # specifies save location of model

    def __init__(self, device="cpu", **model_args):
        super(VideoPredictionModel, self).__init__()
        self.device = device
        self.img_shape = model_args.get("img_shape", (None, None, None))  # h, w, c
        self.img_h, self.img_wm, self.img_c = self.img_shape
        self.action_size = model_args.get("action_size", 0)
        self.action_conditional = model_args.get("action_conditional", False)

        configurable_params = self.config.keys()
        for model_arg in model_args.keys():
            assert model_arg in configurable_params, f"ERROR: encountered invalid model parameter '{model_arg}'. " \
                                                     f"Model '{self.NAME}' supports the following arguments: " \
                                                     f"{configurable_params}"

    @property
    def config(self):
        model_config = {
            "model_dir": self.model_dir,
            "img_shape": self.img_shape,
            "action_size": self.action_size,
            "action_conditional": self.action_conditional,
            "min_context_frames": self.min_context_frames
        }
        return {**model_config, **self._config()}

    def _config(self):
        """ Model-specific config TODO doc"""
        return dict()

    def pred_1(self, x, **kwargs):
        """ Predicts a single frame """
        # input: T frames: [b, T, c, h, w]
        # output: single frame: [b, c, h, w]
        raise NotImplementedError

    def forward(self, x, pred_length=1, **kwargs):
        """ Predicts pred_length frames into the future. """
        # input: T frames: [b, T, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]
        preds = []
        loss_dicts = []
        for i in range(pred_length):
            pred, loss_dict = self.pred_1(x, **kwargs)
            pred = pred.unsqueeze(dim=1)
            preds.append(pred)
            loss_dicts.append(loss_dict)
            x = torch.cat([x[:, 1:], pred], dim=1)

        pred = torch.cat(preds, dim=1)
        if loss_dicts[0] is not None:
            loss_dict = {k: torch.mean([loss_dict[k] for loss_dict in loss_dicts]) for k in loss_dicts[0]}
        else:
            loss_dict = None
        return pred, loss_dict

    def train_iter(self, config, loader, optimizer, loss_provider, epoch):
        """ Default training iteration, traversing the whole loader and TODO """
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
        """ Default evaluation iteration, traversing the whole loader and TODO """
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
