import sys, os, random

sys.path.append(".")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader

from models.vid_pred.copy_last_frame import CopyLastFrameModel
from dataset.synpick_vid import SynpickVideoDataset
from metrics.main import get_prediction_metrics
from utils.visualization import visualize_vid

copy_last_frame_id = "CopyLastFrame baseline"

def test_pred_models(cfg):

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
    data_dir = os.path.join(cfg.data_dir, "test")
    test_data = SynpickVideoDataset(data_dir=data_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                    allow_overlap=cfg.vid_allow_overlap, num_classes=dataset_classes,
                                    include_gripper=cfg.include_gripper)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    iter_loader = iter(test_loader)
    eval_length = len(iter_loader) if cfg.full_test else 10

    if eval_length > 0:
        with torch.no_grad():
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
                    cur_metrics = {**get_prediction_metrics(pred, target, frames=1),
                                   **get_prediction_metrics(pred, target, frames=cfg.vid_pred_length // 2),
                                   **get_prediction_metrics(pred, target)}
                    metric_dicts.append(cur_metrics)

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
                not_first_iter = i > 0
                last_iter = i == len(pm_items) - 1
                wandb.init(config={"full_eval": cfg.full_test, "model": model_desc},
                           project="sem_vp_test_pred", reinit=not_first_iter)
                wandb.log(mean_metric_dict, commit=last_iter)
                if last_iter:
                    wandb.finish()

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