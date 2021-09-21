import lpips
import torch
import torch.nn as nn

# APPLIES TO ALL LOSSES:
# expected shape: [c, h, w]
# expected value range: [-1.0, 1.0]

class LPIPS(nn.Module):
    def __init__(self, device):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='alex') # LPIPS ver. 0.1.4
        self.to(device)

    def forward(self, pred, target):
        pred = list(pred.reshape(-1, *pred.shape[-3:]))
        target = list(target.reshape(-1, *target.shape[-3:]))
        lpips = [self.lpips(t, p) for p, t in zip(pred, target)]
        return torch.mean(torch.stack(lpips, dim=0))