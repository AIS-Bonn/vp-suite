from torch import nn as nn
from torchvision import transforms as TF

from vp_suite.utils.models import ScaleToModel, ScaleToTest


def check_model_and_data_compat(model, dataset, strict_mode=False):
    """
    Checks consistency of given model and given dataset.
    If in strict mode, discrepancies in tensor value range and img size will be bridged with adapters.
    Otherwise, discrepancies are not allowed and will lead to errors.
    """

    model_config = model.config
    dataset_config = dataset.config
    model_preprocessing, model_postprocessing = [], []
    model_dir_str =  f"(location: {model.model_dir})"

    # tensor value range
    model_value_range = list(model_config["tensor_value_range"])
    test_value_range = list(dataset_config["tensor_value_range"])
    if model_value_range != test_value_range:
        if strict_mode:
            raise ValueError(f"Model and run value ranges differ")
        model_preprocessing.append(ScaleToModel(model_value_range, test_value_range))
        model_postprocessing.append(ScaleToTest(model_value_range, test_value_range))

    # img_shape
    model_c, model_h, model_w = model_config["img_shape"]
    test_c, test_h, test_w = dataset_config["img_shape"]
    if model_c != test_c:
        raise ValueError(f"Test dataset provides {test_c}-channel images but "
                         f"Model '{model.NAME}' {model_dir_str} expects {model_c} channels")
    elif model_h != test_h or model_w != test_w:
        if strict_mode:
            raise ValueError(f"Model and run img sizes differ")
        model_preprocessing.append(TF.Resize((model_h, model_w)))
        model_postprocessing.append(TF.Resize((test_h, test_w)))

    # actions
    if model.CAN_HANDLE_ACTIONS and model_config["action_conditional"]:
        if not dataset_config["supports_actions"]:
            raise ValueError("Can't train action-conditional model on a dataset that doesn't provide actions.")
        if model_config["action_size"] != dataset_config["action_size"]:
            raise ValueError("Action size of action-conditional model and dataset must be equal")

    # finalize pre-/postprocessing modules
    model_preprocessing = nn.Sequential(*model_preprocessing)
    model_postprocessing = nn.Sequential(*model_postprocessing)
    return model_preprocessing, model_postprocessing


def check_run_and_model_compat(model, run_config):
    '''
    Checks consistency of the config of a loaded model with given the run configuration.
    '''
    model_config = model.config
    model_dir_str =  f"(location: {model.model_dir})"

    # action conditioning
    mdl_ac, run_ac = model_config["action_conditional"], run_config["use_actions"]
    if model.CAN_HANDLE_ACTIONS:
        if mdl_ac:
            if not run_ac:
                raise ValueError(f"Action-conditioned model '{model.NAME}' {model_dir_str}"
                                 f"can't be invoked without using actions -> set 'use_actions' to True in test cfg!")
        elif run_ac:
            raise ValueError(f"Action-conditionable model '{model.NAME}' {model_dir_str}"
                             f"was trained without using actions -> set 'use_actions' to False in test cfg!")
    elif run_ac:
        print(f"WARNING: Model '{model.NAME}' {model_dir_str} can't handle actions"
              f" -> Testing it without using the actions provided by the dataset")

    # context frames and pred. horizon
    elif run_config["context_frames"] < model.min_context_frames:
        raise ValueError(f"Model '{model.NAME}' {model_dir_str} needs at least "
                         f"{model.min_context_frames} context frames as it uses temporal convolution "
                         f"with said number as kernel size")
