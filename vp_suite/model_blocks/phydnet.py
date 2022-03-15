import math
import numpy as np
import torch
from functools import reduce
from scipy.special import factorial
from torch import nn as nn

from vp_suite.base import VPModelBlock
from vp_suite.model_blocks.conv_lstm_ndrplz import ConvLSTMCell
from vp_suite.model_blocks.conv import DCGANConv, DCGANConvTranspose


class PhyCell_Cell(VPModelBlock):
    r"""
    This class implements a single Cell of the 'PhyCell' component, as introduced in Le Guen and Thome
    (https://arxiv.org/abs/2003.01460) and implemented in https://github.com/vincent-leguen/PhyDNet.
    """
    NAME = "PhyCell - Cell"
    PAPER_REFERENCE = "https://arxiv.org/abs/2003.01460"
    CODE_REFERENCE = "https://github.com/vincent-leguen/PhyDNet"
    MATCHES_REFERENCE = "Not Yet"

    def __init__(self, input_dim, action_conditional, action_size, hidden_dim, kernel_size, bias=True):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.action_size = action_size
        self.action_conditional = action_conditional
        self.F_hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim,
                                             kernel_size=self.kernel_size, stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(find_divisor_for_group_norm(hidden_dim), hidden_dim))
        self.F.add_module('conv2', nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim,
                                             kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=2*self.input_dim, out_channels=self.input_dim,
                                  kernel_size=(3, 3), padding=(1, 1), bias=self.bias)

        if self.action_conditional:
            self.frame_action_conv = nn.Conv2d(in_channels=self.input_dim+self.action_size,
                                               out_channels=self.input_dim, kernel_size=(1, 1))
            self.hidden_action_conv = nn.Conv2d(in_channels=self.input_dim+self.action_size,
                                                out_channels=self.input_dim, kernel_size=(1, 1))

    def forward(self, frame, action, hidden):
        if self.action_conditional:
            inflated_action = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *frame.shape[-2:])
            frame_action = torch.cat([frame, inflated_action], dim=1)  # concatenate along channel axis
            frame = self.frame_action_conv(frame_action)
            hidden_action = torch.cat([hidden, inflated_action], dim=1)  # concatenate along channel axis
            hidden = self.hidden_action_conv(hidden_action)

        combined = torch.cat([frame, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (frame - hidden_tilde)  # correction , Hadamard product
        return next_hidden


class PhyCell(VPModelBlock):
    r"""
    This class implements the 'PhyCell' component, as introduced in Le Guen and Thome
    (https://arxiv.org/abs/2003.01460) and implemented in https://github.com/vincent-leguen/PhyDNet.
    """
    NAME = "PhyCell"
    PAPER_REFERENCE = "https://arxiv.org/abs/2003.01460"
    CODE_REFERENCE = "https://github.com/vincent-leguen/PhyDNet"
    MATCHES_REFERENCE = "Not Yet"

    def __init__(self, input_size, input_dim, hidden_dims, n_layers, kernel_size,
                 action_conditional, action_size, device):
        super(PhyCell, self).__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=self.input_dim,
                                          action_conditional=action_conditional,
                                          action_size=action_size,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, frame, action, first_timestep=False):
        batch_size = frame.data.size()[0]
        if (first_timestep):
            self.init_hidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(frame, action, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], action, self.H[j])
        return self.H, self.H

    def init_hidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_size[0], self.input_size[1]).to(self.device))

    def _set_hidden(self, H):
        self.H = H


class SingleStepConvLSTM(nn.Module):
    r"""
    This class implements a ConvLSTM-Like module used by the PhyDNet introduced in Le Guen and Thome
    (https://arxiv.org/abs/2003.01460) and implemented in https://github.com/vincent-leguen/PhyDNet.
    As opposed to a regular ConvLSTM, this module processes each frame in a separate :meth:`forward()` call.
    """
    def __init__(self, input_size, input_dim, hidden_dims, n_layers, kernel_size,
                 action_conditional, action_size, device):

        super(SingleStepConvLSTM, self).__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.action_size = action_size
        self.action_conditional = action_conditional
        self.device = device

        cell_list = []
        cur_input_dim = self.input_dim + (self.action_size if self.action_conditional else 0)
        for i in range(0, self.n_layers):
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size,
                                          bias=True))
            cur_input_dim = self.hidden_dims[i]
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, frame, action, first_timestep=False):
        batch_size = frame.data.size()[0]
        if first_timestep:
            self.init_hidden(batch_size)  # init Hidden at each forward start

        input = frame
        if self.action_conditional:
            inflated_action = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.input_size)
            input = torch.cat([input, inflated_action], dim=-3)

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)

    def init_hidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_size[0], self.input_size[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_size[0], self.input_size[1]).to(self.device))

    def set_hidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class EncoderSplit(nn.Module):
    r"""
    This class implements an encoder as used by the PhyDNet introduced in Le Guen and Thome
    (https://arxiv.org/abs/2003.01460) and implemented in https://github.com/vincent-leguen/PhyDNet.
    It is similar to the DCGANEncoder introduced in Radford et al. (arxiv.org/abs/1511.06434).
    """
    def __init__(self, in_channels=64, enc_channels=64):
        super(EncoderSplit, self).__init__()
        self.c1 = DCGANConv(in_channels, enc_channels, stride=1)  # (64) x 16 x 16
        self.c2 = DCGANConv(enc_channels, enc_channels, stride=1)  # (64) x 16 x 16

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        return h2


class DecoderSplit(nn.Module):
    r"""
    This class implements a decoder as used by the PhyDNet introduced in Le Guen and Thome
    (https://arxiv.org/abs/2003.01460) and implemented in https://github.com/vincent-leguen/PhyDNet.
    It is similar to the DCGANDecoder introduced in Radford et al. (arxiv.org/abs/1511.06434).
    """
    def __init__(self, out_channels=64, enc_channels=64):
        super(DecoderSplit, self).__init__()
        self.upc1 = DCGANConvTranspose(enc_channels, enc_channels, stride=1)  # (64) x 16 x 16
        self.upc2 = DCGANConvTranspose(enc_channels, out_channels, stride=1)  # (32) x 32 x 32

    def forward(self, x):
        d1 = self.upc1(x)
        d2 = self.upc2(d1)
        return d2


class K2M(nn.Module):
    """
    This class implements methods to convert convolution kernels to moment matrices.
    Used by PhyDNet, which is introduced in Le Guen and Thome
    (https://arxiv.org/abs/2003.01460) and implemented in https://github.com/vincent-leguen/PhyDNet.

    Examples:
        >>> k2m = K2M([5,5])
        >>> k = torch.randn(5,5,dtype=torch.float64)
        >>> m = k2m(k)
    """
    def __init__(self, shape):
        super(K2M, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            M.append(np.zeros((l,l)))
            for i in range(l):
                M[-1][i] = ((np.arange(l)-(l-1)//2)**i)/factorial(i)
            invM.append(np.linalg.inv(M[-1]))
            self.register_buffer('_M'+str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM'+str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        r"""
        Returns: The moment matrix.
        """
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))

    @property
    def invM(self):
        r"""
        Returns: The inverse moment matrix.
        """
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))

    def size(self):
        r"""
        Returns: The kernel size.
        """
        return self._size

    def dim(self):
        r"""
        Returns: The kernel dimensionality.
        """
        return self._dim

    def _packdim(self, x):
        r"""
        TODO
        """
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[np.newaxis,:]
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        return x

    def forward(self, k):
        r"""
        TODO
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k


def _apply_axis_left_dot(x, mats):
    r"""
    TODO
    """
    assert x.dim() == len(mats) + 1
    sizex = x.size()
    k = x.dim() - 1
    for i in range(k):
        x = tensordot(mats[k - i - 1], x, dim=[1, k])
    x = x.permute([k, ] + list(range(k))).contiguous()
    x = x.view(sizex)
    return x


def tensordot(a,b,dim):
    r"""
    TODO Tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x,y:x*y
    if isinstance(dim,int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims
        bdims = [bdims,] if isinstance(bdims, int) else bdims
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims+bdims_
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    a = a.view([-1,N])
    b = b.view([N,-1])
    c = a@b
    return c.view(sizea0+sizeb1)


def find_divisor_for_group_norm(x: int):
    r"""
    Find a good divisor for group norm layer initializations, looking for a value
    that lies close to the square root of the input channel dims.

    Args:
        x (int): Channel dims.

    Returns: The found divisor.
    """
    sq = math.floor(math.sqrt(x))
    while True:
        if x // sq == x / sq:
            return x // sq
        sq -= 1
