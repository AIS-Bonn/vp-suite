from collections import OrderedDict

from vp_suite.model_blocks import ConvLSTM
from vp_suite.models.precipitation_nowcasting.ef_blocks import Encoder_Forecaster


class EF_ConvLSTM(Encoder_Forecaster):
    r"""
    This is a reimplementation of the Encoder-Forecaster model based on ConvLSTMs, as introduced in
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.
    (https://arxiv.org/abs/1506.04214). This implementation is based on the PyTorch implementation on
    https://github.com/Hzzone/Precipitation-Nowcasting which implements the encoder-forecaster structure from
    "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model" by Shi et al.
    (https://arxiv.org/abs/1706.03458).

    The Encoder-Forecaster Network stacks multiple convolutional/up-/downsampling and recurrent layers
    that operate on different spatial scales.

    Note:
        The default hyperparameter configuration is intended for input frames of size (64, 64).
        For considerably larger or smaller image sizes, you might want to adjust the architecture.
    """

    # model-specific constants
    NAME = "EF-ConvLSTM (Shi et al.)"
    PAPER_REFERENCE = "https://arxiv.org/abs/1506.04214"  #: The paper where this model was introduced first.
    CODE_REFERENCE = "https://github.com/Hzzone/Precipitation-Nowcasting"  #: The code location of the reference implementation.
    MATCHES_REFERENCE = "Yes"  #: A comment indicating whether the implementation in this package matches the reference.

    # model hyperparameters (c=channels, h=height, w=width, k=kernel_size, s=stride, p=padding)
    num_layers = 3
    enc_c = [16, 64, 64, 96, 96, 96]  #: Channels for conv and rnn; Length should be 2*num_layers
    dec_c = [96, 96, 96, 96, 64, 16]  #: Channels for conv and rnn; Length should be 2*num_layers

    # convs
    enc_conv_names = ["conv1_leaky_1", "conv2_leaky_1", "conv3_leaky_1"]
    enc_conv_k = [3, 3, 3]
    enc_conv_s = [1, 2, 2]
    enc_conv_p = [1, 1, 1]

    dec_conv_names = ["deconv1_leaky_1", "deconv2_leaky_1", "deconv3_leaky_1"]
    dec_conv_k = [4, 4, 3]
    dec_conv_s = [2, 2, 1]
    dec_conv_p = [1, 1, 1]

    # rnns
    enc_rnn_k = [3, 3, 3]
    enc_rnn_s = [1, 1, 1]
    enc_rnn_p = [1, 1, 1]

    dec_rnn_k = [3, 3, 3]
    dec_rnn_s = [1, 1, 1]
    dec_rnn_p = [1, 1, 1]

    # final convs
    final_conv_1_name = "identity"
    final_conv_1_c = 16
    final_conv_1_k = 3
    final_conv_1_s = 1
    final_conv_1_p = 1

    final_conv_2_name = "conv3_3"
    final_conv_2_k = 1
    final_conv_2_s = 1
    final_conv_2_p = 0

    def __init__(self, device, **model_kwargs):
        super(EF_ConvLSTM, self).__init__(device, **model_kwargs)

    def _build_encoder_decoder(self):
        # build enc layers and encoder
        layer_in_c = self.img_c
        enc_convs, enc_rnns = [], []
        for n in range(self.num_layers):
            layer_mid_c = self.enc_c[2 * n]
            layer_out_c = self.enc_c[2 * n + 1]
            enc_convs.append(OrderedDict(
                {self.enc_conv_names[n]: [layer_in_c, layer_mid_c, self.enc_conv_k[n],
                                          self.enc_conv_s[n], self.enc_conv_p[n]]}
            ))
            enc_rnns.append(ConvLSTM(device=self.device, in_c=layer_mid_c, enc_c=layer_out_c,
                                     state_h=self.enc_rnn_state_h[n], state_w=self.enc_rnn_state_w[n],
                                     kernel_size=self.enc_rnn_k[n], stride=self.enc_rnn_s[n],
                                     padding=self.enc_rnn_p[n]))
            layer_in_c = layer_out_c

        # build dec layers and decoder, including final convs
        dec_convs, dec_rnns = [], []
        for n in range(self.num_layers):
            layer_mid_c = self.dec_c[2 * n]
            layer_out_c = self.dec_c[2 * n + 1]
            dec_rnns.append(ConvLSTM(device=self.device, in_c=layer_in_c, enc_c=layer_mid_c,
                                     state_h=self.dec_rnn_state_h[n], state_w=self.dec_rnn_state_w[n],
                                     kernel_size=self.dec_rnn_k[n], stride=self.dec_rnn_s[n],
                                     padding=self.dec_rnn_p[n]))
            dec_conv_dict = {
                self.dec_conv_names[n]: [layer_mid_c, layer_out_c, self.dec_conv_k[n],
                                         self.dec_conv_s[n], self.dec_conv_p[n]]
            }
            if n == self.num_layers - 1:
                dec_conv_dict[self.final_conv_1_name] = [layer_out_c, self.final_conv_1_c, self.final_conv_1_k,
                                                         self.final_conv_1_s, self.final_conv_1_p]
                dec_conv_dict[self.final_conv_2_name] = [self.final_conv_1_c, self.img_c, self.final_conv_2_k,
                                                         self.final_conv_2_s, self.final_conv_2_p]
            dec_convs.append(OrderedDict(dec_conv_dict))
            layer_in_c = layer_out_c

        return enc_convs, enc_rnns, dec_convs, dec_rnns
