import torch
import torch.nn as nn


class BCELogits(nn.Module):

    def __init__(self):
        super(BCELogits, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        return self.bce(pred, real)