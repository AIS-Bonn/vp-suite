import sys, os, random, json
sys.path.append("")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn
import torchvision.transforms as TF

from vp_suite.models.model_copy_last_frame import CopyLastFrame
from vp_suite.measure.metric_provider import PredictionMetricProvider
from vp_suite.utils.visualization import visualize_vid
from vp_suite.utils.img_processor import ImgProcessor
from vp_suite.dataset.factory import create_test_dataset, update_cfg_from_dataset


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

def get_test_adapters(test_cfg, model_cfg, model):
    '''
    Checks consistency of model configuration with test configuration. Creates appropriate adapter modules
    to make the testing work with that model if the differences can be bridged.
    Some differences (e.g. action-conditioning vs. not) cannot be bridged and will lead to failure.
    '''
    model_preprocessing, model_postprocessing = [], []
    # value range
    model_value_range = list(model_cfg["tensor_value_range"])
    test_value_range = list(test_cfg.tensor_value_range)
    if model_value_range != test_value_range:
        model_preprocessing.append(ScaleToModel(model_value_range, test_value_range))
        model_postprocessing.append(ScaleToTest(model_value_range, test_value_range))

    # action conditioning
    if model.can_handle_actions:
        if model_cfg["use_actions"] != test_cfg.use_actions:
            raise ValueError(f"ERROR: Action-conditioned model '{model.desc}' (loaded from {model_cfg['out_dir']}) "
                             f"can't be invoked without using actions -> set 'use_actions' to True in test cfg!")
        assert model_cfg["action_size"] == test_cfg.action_size,\
            f"ERROR: Action-conditioned model '{model.desc}' (loaded from {model_cfg['out_dir']}) " \
            f"was trained with action size {model_cfg['action_size']}, " \
            f"which is different from the test action size ({test_cfg.action_size})"
    elif test_cfg.use_actions:
        print(f"WARNING: Model '{model.desc}' (loaded from {model_cfg['out_dir']}) can't handle actions"
              f" -> Testing it without using the actions provided by the dataset")

    # img_shape
    model_c, model_h, model_w = model_cfg["img_shape"]
    test_c, test_h, test_w = test_cfg.img_shape
    if model_c != test_c:
        raise ValueError(f"ERROR: Test dataset provides {test_c}-channel images but "
                         f"Model '{model.desc}' (loaded from {model_cfg['out_dir']}) expects {model_c} channels")
    elif model_h != test_h or model_w != test_w:
        model_preprocessing.append(TF.Resize((model_h, model_w)))
        model_postprocessing.append(TF.Resize((test_h, test_w)))

    # TODO are there other model configurations that can be bridged?

    model_preprocessing = nn.Sequential(*model_preprocessing)
    model_postprocessing = nn.Sequential(*model_postprocessing)
    return model_preprocessing, model_postprocessing

def test(test_cfg):

    # prep
    random.seed(test_cfg.seed)
    np.random.seed(test_cfg.seed)
    torch.manual_seed(test_cfg.seed)
    test_cfg.img_processor = ImgProcessor(test_cfg.tensor_value_range)

    # DATASET
    test_data, test_loader = create_test_dataset(test_cfg)
    test_cfg = update_cfg_from_dataset(test_cfg, test_data)

    # MODELS
    models_dict = {}  # desc: (model, model_cfg, preprocessing, postprocessing, test_metrics (initialized empty))
    for model_dir in test_cfg.model_dirs:
        model = torch.load(os.path.join(model_dir, "best_model.pth")).to(test_cfg.device)
        with open(os.path.join(model_dir, "run_cfg.json"), "r") as cfg_file:
            model_cfg = json.load(cfg_file)
        # get adapters to make model work with test cfg
        preprocessing, postprocessing = get_test_adapters(test_cfg, model_cfg, model)
        models_dict[model.desc] = (model, model_cfg, preprocessing, postprocessing, [])

    # add baseline copy model
    clf_baseline = CopyLastFrame().to(test_cfg.device)
    models_dict[clf_baseline.desc] = (clf_baseline, test_cfg, nn.Identity(), nn.Identity(), [])

    iter_loader = iter(test_loader)
    eval_length = 10 if test_cfg.mini_test else len(iter_loader)

    # evaluation / metric calc.
    if eval_length > 0:
        with torch.no_grad():

            metric_provider = PredictionMetricProvider(test_cfg)  # TODO enable config of which metrics to use

            for i in tqdm(range(eval_length)):
                data = next(iter_loader)
                img_data = data["frames"].to(test_cfg.device)
                input = img_data[:, :test_cfg.context_frames]
                target = img_data[:, test_cfg.context_frames:test_cfg.total_frames]
                actions = data["actions"].to(test_cfg.device)

                for (model, model_cfg, preprocess, postprocess, model_metrics_per_dp) in models_dict.values():
                    input = preprocess(input)  # test format to model format
                    if getattr(model, "use_actions", False):
                        pred, _ = model.pred_n(input, pred_length=test_cfg.pred_frames, actions=actions)
                    else:
                        pred, _ = model.pred_n(input, pred_length=test_cfg.pred_frames)
                    pred = postprocess(pred)  # model format to test format
                    cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                    model_metrics_per_dp.append(cur_metrics)

    # save visualizations
    if not test_cfg.no_vis:
        print(f"Saving visualizations for trained models...")
        num_vis = 5
        vis_idx = np.random.choice(len(test_data), num_vis, replace=False)
        for model_desc, (model, model_fp, _) in models_dict.items():
            if model_desc == CopyLastFrame.desc: continue  # don't print for copy baseline
            model_dir = str(Path(model_fp).parent.resolve())
            print(model_desc, model_dir)
            visualize_vid(test_data, test_cfg.context_frames, test_cfg.pred_frames, model, test_cfg.device, test_cfg.img_processor,
                          model_dir, test=True, vis_idx=vis_idx, mode="mp4")

    # log or display metrics
    if eval_length > 0:
        wandb_full_suffix = "" if test_cfg.mini_test else "(full test)"
        models_dict_items = models_dict.items()
        for i, (model_desc, (_, model_fp, model_metrics_per_dp)) in enumerate(models_dict_items):

            # model_metrics_per_dp: list of N lists of F metric dicts (called D).
            # each D_{n, f} contains all calculated metrics for datapoint 'n' and a prediction horizon of 'f' frames.
            # -> Aggregate these metrics over all n, keeping the specific metrics/prediction horizons separate
            datapoint_range = range(len(model_metrics_per_dp))
            frame_range = range(len(model_metrics_per_dp[0]))
            metric_keys = model_metrics_per_dp[0][0].keys()
            mean_metric_dicts = [
                {metric_key: np.mean(
                    [model_metrics_per_dp[dp_i][frame][metric_key] for dp_i in datapoint_range]
                ) for metric_key in metric_keys}
                for frame in frame_range
            ]

            # Log model to WandB
            if not test_cfg.no_wandb:
                print("Logging test results to WandB for all models...")
                wandb.init(config={"mini_test": test_cfg.mini_test, "model_fp": model_fp},
                           project="sem_vp_test_pred", name=f"{model_desc}{wandb_full_suffix}", reinit=(i > 0))
                for f, mean_metric_dict in enumerate(mean_metric_dicts):
                    wandb.log({"pred_frames": f+1, **mean_metric_dict})
                if i == len(models_dict_items) - 1:
                    wandb.finish()

            # Per-model/per-pred.-horizon console log of the metrics
            else:
                print("Printing out test results to terminal...")
                print(f"\n{model_desc} (path: {model_fp}): ")
                for f, mean_metric_dict in enumerate(mean_metric_dicts):
                    print(f"pred_frames: {f + 1}")
                    for (k, v) in mean_metric_dict.items():
                        print(f" -> {k}: {v}")