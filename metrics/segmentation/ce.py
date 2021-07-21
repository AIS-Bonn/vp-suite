import torch
from torch import nn as nn


class CrossEntropyLoss():
    '''
    b means batch_size.
    n means num_classes, including background.
    '''
    def __init__(self):
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def get_loss(self, output: torch.Tensor, target: torch.tensor):

        # assumed shapes [b, n, h, w] for output, [b, 1, h, w] for target
        batch_size, num_classes, h, w = output.shape
        output = output.permute((0, 2, 3, 1)).reshape(-1, num_classes)  # shape: [b*h*w, n+1]
        target = target.view(-1).long()  # [b*h*w]

        return self.loss(output, target)  # scalar, reduced along b*h*w