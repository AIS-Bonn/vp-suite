import torch
import torch.nn as nn


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        mse = self.mse(pred, real)
        return mse.sum(dim=(-1, -2)).mean()