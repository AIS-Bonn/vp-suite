import torch
import torch.nn as nn


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred: torch.Tensor, real : torch.Tensor):
        return self.mse(pred, real)