import math
import numpy as np
import torch
from functools import reduce
from scipy.special import factorial
from torch import nn as nn

from vp_suite.models.model_blocks.conv import DCGANConv, DCGANConvTranspose
from vp_suite.models.model_blocks.enc import DCGANEncoder, DCGANDecoder


class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, action_conditional, action_size, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.action_size = action_size
        self.action_conditional = action_conditional
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(find_divisor_for_group_norm(F_hidden_dim), F_hidden_dim))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=2*self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

        if self.action_conditional:
            self.frame_action_conv = nn.Conv2d(in_channels=self.input_dim+self.action_size,
                                               out_channels=self.input_dim, kernel_size=(1, 1))
            self.hidden_action_conv = nn.Conv2d(in_channels=self.input_dim+self.action_size,
                                               out_channels=self.input_dim, kernel_size=(1, 1))


    def forward(self, frame, action, hidden):  # x [batch_size, hidden_dim, height, width]


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
        next_hidden = hidden_tilde + K * (frame - hidden_tilde)  # correction , Haddamard product

        return next_hidden


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, action_conditional,
                 action_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            print('PhyCell layer ', i, 'input dim ', self.input_dim, ' hidden dim ', self.F_hidden_dims[i])
            cell_list.append(PhyCell_Cell(input_dim=self.input_dim,
                                          action_conditional=action_conditional,
                                          action_size=action_size,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, frame, action, first_timestep=False):  # input_ [batch_size, channels, width, height]
        batch_size = frame.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(frame, action, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], action, self.H[j])

        return self.H, self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H


class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, action_conditional,
                 action_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
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
            print('ConvLSTM layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
            cur_input_dim = self.hidden_dims[i]
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, frame, action, first_timestep=False):  # input_ [batch_size, channels, width, height]
        batch_size = frame.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        input = frame
        if self.action_conditional:
            inflated_action = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.input_shape)
            input = torch.cat([input, inflated_action], dim=-3)

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class EncoderSplit(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(EncoderSplit, self).__init__()
        self.c1 = DCGANConv(nc, nf, stride=1)  # (64) x 16 x 16
        self.c2 = DCGANConv(nf, nf, stride=1)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class DecoderSplit(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(DecoderSplit, self).__init__()
        self.upc1 = DCGANConvTranspose(nf, nf, stride=1)  # (64) x 16 x 16
        self.upc2 = DCGANConvTranspose(nf, nc, stride=1)  # (32) x 32 x 32

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class EncoderRNN(torch.nn.Module):

    def __init__(self, img_shape, phy_cell_channels, phy_kernel_size, action_conditional, action_size, device):
        super(EncoderRNN, self).__init__()
        img_c, _, _ = img_shape
        self.encoder_E = DCGANEncoder(nc=img_c).to(device)
        self.encoder_Ep = EncoderSplit().to(device)
        self.encoder_Er = EncoderSplit().to(device)

        zeros = torch.zeros((1, *img_shape), device=device)
        encoded_zeros = self.encoder_E(zeros)
        self.shape_Ep = self.encoder_Ep(encoded_zeros).shape[1:]
        self.shape_Er = self.encoder_Er(encoded_zeros).shape[1:]

        self.decoder_Dp = DecoderSplit().to(device)
        self.decoder_Dr = DecoderSplit().to(device)
        self.decoder_D = DCGANDecoder(out_size=img_shape[1:], nc=img_c).to(device)

        phy_hidden_dims = [phy_cell_channels, phy_cell_channels, phy_cell_channels]
        self.phycell = PhyCell(input_shape=self.shape_Ep[1:], input_dim=self.shape_Ep[0], F_hidden_dims=phy_hidden_dims,
                               n_layers=3, kernel_size=phy_kernel_size, action_conditional=action_conditional,
                               action_size=action_size, device=device)
        self.phycell.to(device)
        self.convcell = ConvLSTM(input_shape=self.shape_Er[1:], input_dim=self.shape_Ep[0], hidden_dims=[128, 128, 64],
                                 n_layers=3, kernel_size=(3, 3), action_conditional=action_conditional,
                                 action_size=action_size, device=device)
        self.convcell.to(device)
        self.device = device


    def forward(self, frame, action, first_timestep=False, decoding=False):
        frame = self.encoder_E(frame)  # general encoder 64x64x1 -> 32x32x32

        if decoding:  # input=None in decoding phase
            input_phys = None
        else:
            input_phys = self.encoder_Ep(frame)
        input_conv = self.encoder_Er(frame)
        hidden1, output1 = self.phycell(input_phys, action, first_timestep)
        hidden2, output2 = self.convcell(input_conv, action, first_timestep)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = torch.sigmoid(self.decoder_D(concat))
        return out_phys, hidden1, output_image, out_phys, out_conv


class _MK(nn.Module):
    def __init__(self, shape):
        super(_MK, self).__init__()
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
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))
    @property
    def invM(self):
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))

    def size(self):
        return self._size
    def dim(self):
        return self._dim
    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[np.newaxis,:]
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass


def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats) + 1
    sizex = x.size()
    k = x.dim() - 1
    for i in range(k):
        x = tensordot(mats[k - i - 1], x, dim=[1, k])
    x = x.permute([k, ] + list(range(k))).contiguous()
    x = x.view(sizex)
    return x


class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """
    def __init__(self, shape):
        super(K2M, self).__init__(shape)
    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k

def tensordot(a,b,dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
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


def find_divisor_for_group_norm(x):
    sq = math.floor(math.sqrt(x))
    while True:
        if x // sq == x / sq: return x // sq
        sq -= 1