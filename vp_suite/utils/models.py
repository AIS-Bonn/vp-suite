from typing import List, Union, Tuple

import torch
from torch import nn as nn


class ScaleToTest(nn.Module):
    r"""
    This class acts as an adapter module that scales pixel values from the model domain to the test run domain.
    """
    def __init__(self, model_value_range: List[float], test_value_range: List[float]):
        r"""
        Initializes the scaler module by setting the model domain and test domain value range.

        Args:
            model_value_range (List[float]): The model's value range.
            test_value_range (List[float]): The test run's value range.
        """
        super(ScaleToTest, self).__init__()
        self.m_min, self.m_max = model_value_range
        self.t_min, self.t_max = test_value_range

    def forward(self, img : torch.Tensor):
        r"""
        Scales the input image from the model domain to the test run domain.

        Args:
            img (torch.Tensor): The image to scale.

        Returns: The scaled image.
        """
        img = (img - self.m_min) / (self.m_max - self.m_min)  # [0., 1.]
        img = img * (self.t_max - self.t_min) + self.t_min  # [test_val_min, test_val_max]
        return img


class ScaleToModel(nn.Module):
    r"""
    This class acts as an adapter module that scales pixel values from the test run domain to the model domain.
    """
    def __init__(self, model_value_range, test_value_range):
        r"""
        Initializes the scaler module by setting the model domain and test domain value range.

        Args:
            model_value_range (List[float]): The model's value range.
            test_value_range (List[float]): The test run's value range.
        """
        super(ScaleToModel, self).__init__()
        self.m_min, self.m_max = model_value_range
        self.t_min, self.t_max = test_value_range

    def forward(self, img: torch.Tensor):
        r"""
        Scales the input image from the test run domain to the model domain.

        Args:
            img (torch.Tensor): The image to scale.

        Returns: The scaled image.
        """
        img = (img - self.t_min) / (self.t_max - self.t_min)  # [0., 1.]
        img = img * (self.m_max - self.m_min) + self.m_min  # [model_val_min, model_val_max]
        return img


def state_dicts_equal(model1: nn.Module, model2: nn.Module,
                      check_values: bool = False, verbose: bool = False):
    r"""
    Checks whether two models are equal with respect to their state dicts.
    Modified from: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212

    Args:
        model1 (nn.Module): Model 1.
        model2 (nn.Module): Model 2.
        check_values (bool): If specified, also compares the values of the state dicts. By default, only the keys and
        dimensionalities are checked
        verbose (bool): If specified, prints all state dict components to console

    Returns: True if both state dicts are equal in keys and values, False (with debug prints) otherwise.
    """
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    if verbose:
        for param_tensor in model1_state_dict:
            print(param_tensor, "\t", model1_state_dict[param_tensor].size())
        print("")
        for param_tensor in model2_state_dict:
            print(param_tensor, "\t", model2_state_dict[param_tensor].size())

    if len(model1_state_dict) != len(model2_state_dict):
        print(
            f"Length mismatch: model1 {len(model1_state_dict)}, model2 {len(model2_state_dict)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model1_state_dict.keys())).startswith("module"):
        model1_state_dict = {
            k[len("module") + 1 :]: v for k, v in model1_state_dict.items()
        }

    if next(iter(model2_state_dict.keys())).startswith("module"):
        model2_state_dict = {
            k[len("module") + 1 :]: v for k, v in model2_state_dict.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model1_state_dict.items(), model2_state_dict.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if v_1.shape != v_2.shape:
            print(f"Tensor shape mismatch: model1 {k_1} is {v_1.shape}, model2 {k_2} is {v_2.shape}")
            return False

        if check_values and not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: model1 values of '{k_1}' vs model2 values of '{k_2}'")
            return False
    return True


def conv_output_shape(h_w: Union[int, Tuple[int]], kernel_size=1, stride=1, pad=0, dilation=1):
    """
    SOURCE: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    Utility function for computing output size of convolutions given the input size and the conv layer parameters.

    Args:
        h_w (Union[int, Tuple[int]]): The input height and width, either as a single integer number or as a tuple.
        kernel_size (int): The layer's kernel size.
        stride (int): The layer's stride.
        pad (int): The layer's padding.
        dilation (int): The layer's dilation.

    Returns: A tuple (height, width) with the resulting height and width after layer application.
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    SOURCE: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    Utility function for computing output size of convTransposes given the input size and the convT layer parameters.

    Args:
        h_w (Union[int, Tuple[int]]): The input height and width, either as a single integer number or as a tuple.
        kernel_size (int): The layer's kernel size.
        stride (int): The layer's stride.
        pad (int): The layer's padding.
        dilation (int): The layer's dilation.

    Returns: A tuple (height, width) with the resulting height and width after layer application.
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + (kernel_size[0] - 1) + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + (kernel_size[1] - 1) + pad[1]

    return h, w
