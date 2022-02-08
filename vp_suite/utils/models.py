import torch
from torch import nn as nn


class ScaleToTest(nn.Module):
    def __init__(self, model_value_range, test_value_range):
        super(ScaleToTest, self).__init__()
        self.m_min, self.m_max = model_value_range
        self.t_min, self.t_max = test_value_range

    def forward(self, img : torch.Tensor):
        ''' input: [model_val_min, model_val_max] '''
        img = (img - self.m_min) / (self.m_max - self.m_min)  # [0., 1.]
        img = img * (self.t_max - self.t_min) + self.t_min  # [test_val_min, test_val_max]
        return img


class ScaleToModel(nn.Module):
    def __init__(self, model_value_range, test_value_range):
        super(ScaleToModel, self).__init__()
        self.m_min, self.m_max = model_value_range
        self.t_min, self.t_max = test_value_range

    def forward(self, img: torch.Tensor):
        ''' input: [test_val_min, test_val_max] '''
        img = (img - self.t_min) / (self.t_max - self.t_min)  # [0., 1.]
        img = img * (self.m_max - self.m_min) + self.m_min  # [model_val_min, model_val_max]
        return img


def state_dicts_equal(model1: nn.Module, model2: nn.Module, check_values: bool = False):
    r"""
    Checks whether two models are equal with respect to their state dicts.
    Modified from: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
    Args:
        model1 (nn.Module): Model 1.
        model2 (nn.Module): Model 2.
        check_values (bool): If specified, also compares the values of the state dicts. By default, only the keys and
        dimensionalities are checked

    Returns: True if both state dicts are equal in keys and values, False (with debug prints) otherwise.
    """
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    #for param_tensor in model1_state_dict:
    #    print(param_tensor, "\t", model1_state_dict[param_tensor].size())
    #print("")
    #for param_tensor in model2_state_dict:
    #    print(param_tensor, "\t", model2_state_dict[param_tensor].size())

    if len(model1_state_dict) != len(model2_state_dict):
        print(
            f"Length mismatch: {len(model1_state_dict)}, {len(model2_state_dict)}"
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
            print(f"Tensor shape mismatch: {k_1} is {v_1.shape}, {k_2} is {v_2.shape}")
            return False

        if check_values and not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: values of '{k_1}' vs values of '{k_2}'")
            return False
    return True