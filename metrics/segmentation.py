import torch

# APPLIES TO ALL METRICS:
# expected shape: [h, w]
# expected value range: class labels starting from 0


def Accuracy(pred, target):
    '''
    input type: torch.tensor (torch.int)
    '''
    num_pixels = torch.numel(pred)
    num_correct = (pred == target).sum()
    return 100.0 * num_correct / num_pixels