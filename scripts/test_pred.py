import sys, random
from typing import List

sys.path.append(".")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch

from models.vid_pred.pred_model import VideoPredictionModel
from models.vid_pred.copy_last_frame import CopyLastFrameModel
from metrics.main import PredictionMetricProvider
from utils.visualization import visualize_vid
from dataset.dataset import create_dataset


def test_pred_models(cfg, test_data_and_loader=None):

    # prep
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset_classes = cfg.dataset_classes+1 if cfg.include_gripper else cfg.dataset_classes

    # MODELS
    model_fps = cfg.models
    pred_models : List[VideoPredictionModel] = [torch.load(model_path).to(cfg.device) for model_path in model_fps]
    if cfg.program == "test_pred":
        pred_models.append(CopyLastFrameModel().to(cfg.device))  # add baseline copy model
        model_fps.append("")
    models_dict = {model.desc: (model, fp, []) for model, fp in zip(pred_models, model_fps)}

    # DATASET
    if test_data_and_loader is None:
        (_, _, test_data), (_, _, test_loader) = create_dataset(cfg)
    else:
        test_data, test_loader = test_data_and_loader

    iter_loader = iter(test_loader)
    eval_length = len(iter_loader) if cfg.full_test else 10

    # evaluation / metric calc.
    if eval_length > 0:
        with torch.no_grad():

            metric_provider = PredictionMetricProvider(cfg)

            for i in tqdm(range(eval_length)):
                data = next(iter_loader)
                img_data = data[cfg.pred_mode].to(cfg.device)
                input = img_data[:, :cfg.vid_input_length]
                target = img_data[:, cfg.vid_input_length:cfg.vid_total_length]
                actions = data["actions"].to(cfg.device)

                for (model, _, model_metrics_per_dp) in models_dict.values():
                    if getattr(model, "use_actions", False):
                        pred, _ = model.pred_n(input, pred_length=cfg.vid_pred_length, actions=actions)
                    else:
                        pred, _ = model.pred_n(input, pred_length=cfg.vid_pred_length)
                    cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                    model_metrics_per_dp.append(cur_metrics)

    # save visualizations
    if not cfg.no_vis:
        print(f"Saving visualizations for trained models...")
        num_channels = dataset_classes if cfg.pred_mode == "mask" else 3
        num_vis = 5
        vis_idx = np.random.choice(len(test_data), num_vis, replace=False)
        for model_desc, (model, model_fp, _) in models_dict.items():
            if model_desc == CopyLastFrameModel.desc: continue  # don't print for copy baseline
            model_dir = str(Path(model_fp).parent.resolve())
            print(model_desc, model_dir)
            visualize_vid(test_data, cfg.vid_input_length, cfg.vid_pred_length, model, cfg.device, model_dir,
                          (cfg.pred_mode, num_channels), test=True, vis_idx=vis_idx, mode="mp4")

    # log or display metrics
    if eval_length > 0:
        wandb_full_suffix = " (full test)" if cfg.full_test else ""
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

            # return metrics for the first model if called from another program
            if cfg.program != "test_pred":
                return mean_metric_dicts

            # Log model to WandB
            elif not cfg.no_wandb:
                print("Logging test results to WandB for all models...")
                wandb.init(config={"full_eval": cfg.full_test, "model_fp": model_fp},
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
