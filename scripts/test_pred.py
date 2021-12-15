import sys, os, random

sys.path.append(".")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch

from models.vid_pred.copy_last_frame import CopyLastFrameModel
from metrics.main import PredictionMetricProvider
from utils.visualization import visualize_vid
from dataset.dataset import create_dataset


copy_last_frame_id = "CopyLastFrame baseline"

def test_pred_models(cfg, test_stuff=None):

    # prep
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset_classes = cfg.dataset_classes+1 if cfg.include_gripper else cfg.dataset_classes

    # MODELS
    pred_models = {model_path: (torch.load(model_path).to(cfg.device), []) for model_path in cfg.models}
    if cfg.program == "test_pred":
        pred_models[copy_last_frame_id] = (CopyLastFrameModel().to(cfg.device), [])

    # DATASET
    if test_stuff is None:
        (_, _, test_data), (_, _, test_loader) = create_dataset(cfg)
    else:
        test_data, test_loader = test_stuff

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

                for (model, metric_dicts) in pred_models.values():
                    if getattr(model, "use_actions", False):
                        pred, _ = model.pred_n(input, pred_length=cfg.vid_pred_length, actions=actions)
                    else:
                        pred, _ = model.pred_n(input, pred_length=cfg.vid_pred_length)
                    cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                    metric_dicts.append(cur_metrics)

    # save visualizations
    if not cfg.no_vis:
        print(f"Saving visualizations (except for {copy_last_frame_id})...")
        num_channels = dataset_classes if cfg.pred_mode == "mask" else 3
        num_vis = 5
        vis_idx = np.random.choice(len(test_data), num_vis, replace=False)
        for model_path, (model, _) in pred_models.items():
            if model_path != copy_last_frame_id:
                model_dir = str(Path(model_path).parent.resolve())
                print(model_path, model_dir)
                visualize_vid(test_data, cfg.vid_input_length, cfg.vid_pred_length, model, cfg.device, model_dir,
                              (cfg.pred_mode, num_channels), test=True, vis_idx=vis_idx, mode="mp4")

    # log or display metrics
    if eval_length > 0:
        pm_items = pred_models.items()
        for i, (model_desc, (_, metric_dicts)) in enumerate(pm_items):
            mean_metric_dict = {k: np.mean([m_dict[k] for m_dict in metric_dicts]) for k in metric_dicts[0].keys()}
            print(f"\n{model_desc}: ")
            for (k, v) in mean_metric_dict.items():
                print(f"{k}: {v}")

            # optuna is used -> return metrics for optimization.
            # otherwise, log to WandB if desired.
            if cfg.program != "test_pred":
                return mean_metric_dict
            elif not cfg.no_wandb:
                wandb.init(config={"full_eval": cfg.full_test, "model": model_desc},
                           project="sem_vp_test_pred", reinit=(i > 0))
                for frame_cnt in range(1, cfg.vid_pred_length + 1):
                    fstr = str(frame_cnt)
                    frame_cnt_dict = {k.replace("_" + fstr, ''): v for k, v in mean_metric_dict.items() if fstr in k}
                    wandb.log({**frame_cnt_dict, "pred. frames": frame_cnt}, commit=True)
                if i == len(pm_items) - 1:
                    wandb.finish()