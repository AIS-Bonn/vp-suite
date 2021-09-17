import sys, os, argparse
sys.path.append(".")

from pathlib import Path
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.prediction.copy_last_frame import CopyLastFrameModel
from dataset import SynpickVideoDataset, postprocess_mask, postprocess_img, preprocess_img, preprocess_mask_inflate
from metrics.main import get_prediction_metrics
from visualize import visualize_vid

copy_last_frame_id = "CopyLastFrame baseline"

def test_pred_models(cfg):

    # prep
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset_classes = cfg.dataset_classes+1 if cfg.include_gripper else cfg.dataset_classes

    # MODELS
    pred_models = {model_path: (torch.load(model_path).to(cfg.device), []) for model_path in cfg.models}
    pred_models[copy_last_frame_id] = (CopyLastFrameModel().to(cfg.device), [])

    # DATASET
    data_dir = os.path.join(cfg.data_dir, "test")
    test_data = SynpickVideoDataset(data_dir=data_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                    allow_overlap=cfg.vid_allow_overlap, num_classes=dataset_classes,
                                    include_gripper=cfg.include_gripper)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    iter_loader = iter(test_loader)
    eval_length = len(iter_loader) if cfg.full_test else 10

    with torch.no_grad():
        for i in tqdm(range(eval_length)):
            data = next(iter_loader)
            img_data = data[cfg.pred_mode].to(cfg.device)
            input = img_data[:, :cfg.vid_in_length]
            target = img_data[:, cfg.vid_in_length:cfg.vid_total_length]
            actions = data["actions"].to(cfg.device)

            for (model, metric_dicts) in pred_models.values():
                if getattr(model, "use_actions", False):
                    pred, _ = model.pred_n(input, pred_length=cfg.vid_pred_length, actions=actions)
                else:
                    pred, _ = model.pred_n(input, pred_length=cfg.vid_pred_length)
                metric_dicts.append(get_prediction_metrics(pred, target))

    for model_desc, (_, metric_dicts) in pred_models.items():
        mean_metric_dict = {k: np.mean([m_dict[k] for m_dict in metric_dicts]) for k in metric_dicts[0].keys()}
        print(f"\n{model_desc}: ")
        for (k, v) in mean_metric_dict.items():
            print(f"{k}: {v}")

    print(f"Saving visualizations (except for {copy_last_frame_id})...")
    num_channels = dataset_classes if cfg.pred_mode == "mask" else 3
    for model_path, (model, _) in pred_models.items():
        if model_path != copy_last_frame_id:
            model_dir = str(Path(model_path).parent.resolve())
            print(model_path, model_dir)
            visualize_vid(test_data, cfg.vid_in_length, cfg.vid_pred_length, model, cfg.device, model_dir,
                          (cfg.pred_mode, num_channels), num_vis=5, test=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Prediction Evaluation")
    parser.add_argument("--data-dir", type=str, help="Path to data dir")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--pred-mode", type=str, choices=["rgb", "colorized", "mask"], default="rgb",
                        help="Which kind of data to test on")
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--full-evaluation", action="store_true", help="If specified, checks the whole test set")
    parser.add_argument("--models", nargs="*", type=str, default=[], help="Paths to prediction models")

    cfg = parser.parse_args()
    test_pred_models(cfg)