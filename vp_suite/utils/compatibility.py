from torch import nn as nn
from torchvision import transforms as TF

from vp_suite.utils.models import ScaleToModel, ScaleToTest


def check_model_and_data_compat(model_config, dataset_config, model):
    """ TODO doc """

    # img shape
    model_img_shape = model_config.get("img_shape", dataset_config["img_shape"])
    assert model_img_shape== dataset_config["img_shape"], \
        "expected img shape of loaded model and dataset img_shape to be the same"

    # actions
    if model.can_handle_actions and model_config["action_conditional"]:
        assert dataset_config["supports_actions"], \
            "ERROR: can't train action-conditional model on a dataset that doesn't provide actions."
        assert model_config["action_size"] == dataset_config["action_size"], \
            "ERROR: action size of action-conditional model and dataset must be equal"


def check_run_and_model_compat(model_config, run_config, model, strict_mode=False, model_dir:str=None):
    '''
    Checks consistency of the config of a loaded model with given the run configuration.
    Creates appropriate adapter modules to bridge the differences if possible and not in strict_mode.
    Some differences (e.g. action-conditioning vs. not) cannot be bridged and will lead to failure.
    If strict_mode is active, strict compatibility is enforced (adapters are not allowed)
    '''
    model_preprocessing, model_postprocessing = [], []
    model_origin_str =  "" if model_dir is None else f"(loaded from {model_dir})"

    # value range
    model_value_range = list(model_config["tensor_value_range"])
    test_value_range = list(run_config["tensor_value_range"])
    if model_value_range != test_value_range:
        if strict_mode:
            raise ValueError(f"ERROR: model and run value ranges differ")
        model_preprocessing.append(ScaleToModel(model_value_range, test_value_range))
        model_postprocessing.append(ScaleToTest(model_value_range, test_value_range))

    # action conditioning
    mdl_ac, run_ac = model_config["use_actions"], run_config["use_actions"]
    if model.can_handle_actions:
        if mdl_ac:
            if not run_ac:
                raise ValueError(f"ERROR: Action-conditioned model '{model.desc}' {model_origin_str}"
                                 f"can't be invoked without using actions -> set 'use_actions' to True in test cfg!")
            else:
                assert model_config["action_size"] == run_config["action_size"], \
                    f"ERROR: Action-conditioned model '{model.desc}' {model_origin_str} " \
                    f"was trained with action size {model_config['action_size']}, " \
                    f"which is different from the dataset's action size ({run_config['action_size']})"
        elif run_ac:
            raise ValueError(f"ERROR: Action-conditionable model '{model.desc}' {model_origin_str}"
                             f"was trained without using actions -> set 'use_actions' to False in test cfg!")
    elif run_ac:
        print(f"WARNING: Model '{model.desc}' {model_origin_str} can't handle actions"
              f" -> Testing it without using the actions provided by the dataset")

    # img_shape
    model_c, model_h, model_w = model_config["img_shape"]
    test_c, test_h, test_w = run_config["img_shape"]
    if model_c != test_c:
        raise ValueError(f"ERROR: Test dataset provides {test_c}-channel images but "
                         f"Model '{model.desc}' {model_origin_str} expects {model_c} channels")
    elif model_h != test_h or model_w != test_w:
        if strict_mode:
            raise ValueError(f"ERROR: model and run img sizes differ")
        model_preprocessing.append(TF.Resize((model_h, model_w)))
        model_postprocessing.append(TF.Resize((test_h, test_w)))

    # context frames and pred. horizon
    if run_config["context_frames"] is None:
        run_config["context_frames"] = model_config["context_frames"]
    elif run_config["context_frames"] < model.min_context_frames:
        raise ValueError(f"ERROR: Model '{model.desc}' {model_origin_str} needs at least "
                         f"{model.min_context_frames} context frames as it uses temporal convolution "
                         f"with said number as kernel size")
    if run_config["pred_frames"] is None:
        run_config["pred_frames"] = model_config["pred_frames"]

    # finalize pre-/postprocessing modules
    model_preprocessing = nn.Sequential(*model_preprocessing)
    model_postprocessing = nn.Sequential(*model_postprocessing)
    return model_preprocessing, model_postprocessing