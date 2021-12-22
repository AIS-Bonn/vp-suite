import torch
import torch.nn as nn


class VideoPredictionModel(nn.Module):

    trainable = True  # most implemented models will be trainable
    can_handle_actions = False  # models by default won't be able to handle actions

    def __init__(self, cfg):
        super(VideoPredictionModel, self).__init__()
        if cfg is not None:
            self.img_shape = cfg.img_shape
            self.img_c, self.img_h, self.img_w = self.img_shape
            self.action_size = cfg.action_size
            self.use_actions = self.action_size > 0 and cfg.use_actions
            self.device = cfg.device

    @classmethod
    def model_desc(cls):
        raise NotImplementedError

    @property
    def desc(self):
        return self.__class__.model_desc()


    def forward(self, x, **kwargs):
        # input: T frames: [b, T, c, h, w]
        # output: single frame: [b, c, h, w]
        raise NotImplementedError

    def pred_n(self, x, pred_length=1, **kwargs):
        # input: T frames: [b, T, c, h, w]
        # output: pred_length (P) frames: [b, P, c, h, w]
        preds = []
        loss_dicts = []
        for i in range(pred_length):
            pred, loss_dict = self.forward(x, **kwargs)
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