import sys, os, argparse
sys.path.append(".")

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.prediction.copy_last_frame import CopyLastFrameModel
from dataset import SynpickVideoDataset, postprocess_mask, postprocess_img, preprocess_img, preprocess_mask_inflate
from config import *
from metrics.main import get_prediction_metrics


def evaluate_pred_models(cfg):

    # prep
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset_classes = SYNPICK_CLASSES+1 if cfg.include_gripper else SYNPICK_CLASSES

    # MODELS
    pred_models = {model_path: (torch.load(model_path).to(DEVICE), []) for model_path in cfg.models}
    pred_models["CopyLastFrame baseline"] = (CopyLastFrameModel().to(DEVICE), [])

    # DATASET
    data_dir = os.path.join(cfg.data_dir, "test")
    test_data = SynpickVideoDataset(data_dir=data_dir, num_frames=VIDEO_TOT_LENGTH, step=VID_STEP,
                                    allow_overlap=VID_DATA_ALLOW_OVERLAP, num_classes=dataset_classes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    iter_loader = iter(test_loader)
    eval_length = len(iter_loader) if cfg.full_evaluation else 10

    with torch.no_grad():
        for i in tqdm(range(eval_length)):
            data = next(iter_loader)
            img_data = data[cfg.pred_mode].to(DEVICE)
            input = img_data[:, :VIDEO_IN_LENGTH]
            target = img_data[:, VIDEO_IN_LENGTH:VIDEO_IN_LENGTH+VIDEO_PRED_LENGTH]
            actions = data["actions"].to(DEVICE)

            for (model, metric_dicts) in pred_models.values():
                if getattr(model, "use_actions", False):
                    pred, _ = model.pred_n(input, pred_length=VIDEO_PRED_LENGTH, actions=actions)
                else:
                    pred, _ = model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)
                metric_dicts.append(get_prediction_metrics(pred, target))

    for model_desc, (_, metric_dicts) in pred_models.items():
        mean_metric_dict = {k: np.mean([m_dict[k] for m_dict in metric_dicts]) for k in metric_dicts[0].keys()}
        print(f"\n{model_desc}: ")
        for (k, v) in mean_metric_dict.items():
            print(f"{k}: {v}")

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
    evaluate_pred_models(cfg)