from torch import nn
import torch

from vp_suite.base import VPModelBlock


class ConvLSTM(VPModelBlock):
    r"""
    This class implements a convolutional LSTM, as introduced in Shi et al. (https://arxiv.org/abs/1506.04214) and
    implemented in https://github.com/Hzzone/Precipitation-Nowcasting. This is the 'original' ConvLSTM.
    """
    NAME = "ConvLSTM (Shi et al.)"
    PAPER_REFERENCE = "https://arxiv.org/abs/1506.04214"
    CODE_REFERENCE = "https://github.com/Hzzone/Precipitation-Nowcasting"
    MATCHES_REFERENCE = "Yes"

    def __init__(self, device, in_channels, enc_channels, state_h, state_w, kernel_size, stride=1, padding=1):
        super().__init__()
        self.device = device
        self._conv = nn.Conv2d(in_channels=in_channels + enc_channels,
                               out_channels=enc_channels * 4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.state_h = state_h
        self.state_w = state_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # However, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, enc_channels, self.state_h, self.state_w)).to(self.device)
        self.Wcf = nn.Parameter(torch.zeros(1, enc_channels, self.state_h, self.state_w)).to(self.device)
        self.Wco = nn.Parameter(torch.zeros(1, enc_channels, self.state_h, self.state_w)).to(self.device)
        self.in_c = in_channels
        self.enc_c = enc_channels

    # inputs and states should not be all none
    # inputs: [b, t, c, h, w]
    def forward(self, inputs, states, seq_len):

        if states is None:
            b = inputs.shape[0]
            c = torch.zeros((b, self.enc_c, self.state_h, self.state_w),
                            dtype=torch.float, device=self.device)
            h = torch.zeros((b, self.enc_c, self.state_h, self.state_w),
                            dtype=torch.float, device=self.device)
        else:
            h, c = states
            b = h.shape[0]
        T = seq_len

        outputs = []
        for t in range(T):
            # initial inputs
            if inputs is None:
                x = torch.zeros((b, self.in_c, self.state_h,
                                 self.state_w), dtype=torch.float, device=self.device)
            else:
                x = inputs[:, t]  # mustn't be None. Should be zero on first decoder step
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)

