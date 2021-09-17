import sys, os, argparse
sys.path.append(".")

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SynpickVideoDataset, postprocess_mask, postprocess_img, preprocess_img, preprocess_mask_inflate
from utils import colorize_semseg, save_vid_vis

def visualize_4_way(cfg):

    # SEEDING
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # MODELS
    if len(cfg.models) != 4:
        raise ValueError("4way_vis expects 4 models in this order: [seg, pred_rgb, pred_mask, pred_colorized]")
    seg_model, pred_rgb_model, pred_mask_model, pred_colorized_model \
        = [torch.load(model_path) for model_path in cfg.models]
    seg_model.eval()
    pred_rgb_model.eval()
    pred_mask_model.eval()
    pred_colorized_model.eval()

    dataset_classes = cfg.dataset_classes+1 if cfg.include_gripper else cfg.dataset_classes

    # DATASET
    data_dir = os.path.join(cfg.data_dir, "test")
    test_data = SynpickVideoDataset(data_dir=data_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                    allow_overlap=cfg.vid_allow_overlap, num_classes=dataset_classes,
                                    include_gripper=cfg.include_gripper)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    iter_loader = iter(test_loader)

    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)

    # METRICS
    def mse(pred, target):
        return np.square(np.subtract(pred, target)).mean()

    mse_seg_only, mse_pred_rgb, mse_pred_mask, mse_pred_colorized = [], [], [], []
    eval_length = len(iter_loader) if cfg.full_evaluation else 10

    with torch.no_grad():
        for i in tqdm(range(eval_length)):

            data = next(iter_loader)
            imgs, colorized_masks, actions \
                = data["rgb"].to(cfg.device), data["colorized"].to(cfg.device), data["actions"].to(cfg.device)  # [1, T, 3, h, w]
            
            gt_rgb_vis = postprocess_img(imgs.squeeze(dim=0))  # [T, h, w, 3]
            gt_colorized_vis = postprocess_img(colorized_masks.squeeze(dim=0))  # [T, h, w, 3]
            input = imgs[:, :cfg.vid_in_length]  # [1, t, 3, h, w]

            pred_rgb, _ = pred_rgb_model.pred_n(input, pred_length=cfg.vid_pred_length, actions=actions)
            pred_rgb = torch.cat([input, pred_rgb], dim=1)  # [1, T, 3, h, w]
            pred_rgb_vis = postprocess_img(pred_rgb.squeeze(dim=0))  # [T, 3, h, w]

            pred_then_seg = torch.stack([seg_model(pred_rgb[:, i]) for i in range(pred_rgb.shape[1])], dim=1)
            pred_then_seg = pred_then_seg.argmax(dim=2).squeeze()  # [T, h, w]
            pred_seg_color_vis = colorize_semseg(postprocess_mask(pred_then_seg), num_classes=dataset_classes).transpose(0, 3, 1, 2) # [T, 3, h, w]

            seg = torch.stack([seg_model(imgs[:, i]) for i in range(imgs.shape[1])], dim=1).argmax(dim=2)  # [1, T, 1, h, w]
            seg_input = torch.stack([(seg == i) for i in range(dataset_classes)], dim=2).float()  # [1, T, c, h, w] one-hot float
            input_seg = seg_input[:, :cfg.vid_in_length]  # [1, t, c, h, w]
            seg_then_pred, _ = pred_mask_model.pred_n(input_seg, pred_length=cfg.vid_pred_length, actions=actions)
            seg_then_pred = seg_then_pred.argmax(dim=2)  # [1, n, 1, h, w]
            seg_then_pred = torch.cat([input_seg.argmax(dim=2), seg_then_pred], dim=1).squeeze()  # [T, h, w]
            seg_pred_color_vis = colorize_semseg(postprocess_mask(seg_then_pred), num_classes=dataset_classes).transpose(0, 3, 1, 2)  # [T, 3, h, w]

            seg_colorized = colorize_semseg(postprocess_mask(seg.squeeze()), num_classes=dataset_classes)
            seg_color_per_frame_vis = seg_colorized.transpose(0, 3, 1, 2)  # [T, 3, h, w]

            input_colorized = preprocess_img(seg_colorized[:cfg.vid_in_length]).to(cfg.device).unsqueeze(dim=0)  # [b, t, 3, h, w]
            seg_color_pred, _ = pred_colorized_model.pred_n(input_colorized, pred_length=cfg.vid_pred_length, actions=actions)
            seg_color_pred = torch.cat([input_colorized, seg_color_pred], dim=1).squeeze(dim=0)
            seg_color_pred_vis = postprocess_img(seg_color_pred)  # [T, 3, h, w]

            mse_seg_only.append(mse(seg_color_per_frame_vis, gt_colorized_vis))
            mse_pred_rgb.append(mse(pred_seg_color_vis, gt_colorized_vis))
            mse_pred_mask.append(mse(seg_pred_color_vis, gt_colorized_vis))
            mse_pred_colorized.append(mse(seg_color_pred_vis, gt_colorized_vis))

            save_vid_vis(
                out_fp=os.path.join(cfg.out_dir, "4way_vis_{}.gif".format(str(i))),
                video_in_length=cfg.vid_in_length,
                True_Trajectory_RGB=gt_rgb_vis,
                True_Trajectory_Seg=gt_colorized_vis,
                Framewise_Segmentation=seg_color_per_frame_vis,
                RGB_Prediction=pred_rgb_vis,
                RGB_Prediction_Colorized=pred_seg_color_vis,
                Mask_Prediction_Colorized=seg_pred_color_vis,
                Colorization_Prediction=seg_color_pred_vis
            )

    data = [np.array(mse_seg_only), np.array(mse_pred_rgb), np.array(mse_pred_mask), np.array(mse_pred_colorized)]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['seg. only', 'pred. on RGB',
                        'pred. on mask', 'pred. on colorized'])
    bp = ax.boxplot(data, showmeans=True, meanline=True, notch=True)
    plt.title("MSE values")
    plt.savefig(os.path.join(cfg.out_dir, "mse_values.jpg"))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Video Prediction 4way vis")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--seg", type=str, help="Path to segmentation model")
    parser.add_argument("--pred-rgb", type=str, help="Path to prediction model (rgb)")
    parser.add_argument("--pred-mask", type=str, help="Path to prediction model (masks)")
    parser.add_argument("--pred-colorized", type=str, help="Path to prediction model (colorized)")
    parser.add_argument("--data-dir", type=str, help="Path to data dir")
    parser.add_argument("--out-dir", type=str, help="Output path for results")
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--full-evaluation", action="store_true", help="If specified, checks the whole test set")

    cfg = parser.parse_args()
    visualize_4_way(cfg)