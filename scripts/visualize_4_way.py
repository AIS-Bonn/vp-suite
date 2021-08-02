import sys, os, argparse
sys.path.append(".")

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SynpickVideoDataset, postprocess_mask, postprocess_img, preprocess_img, preprocess_mask_inflate
from config import *
from utils import colorize_semseg, save_vid_vis

def visualize_4_way(cfg):

    # MODELS
    seg_model = torch.load(cfg.seg)
    pred_rgb_model = torch.load(cfg.pred_rgb)
    pred_mask_model = torch.load(cfg.pred_mask)
    pred_colorized_mask_model = torch.load(cfg.pred_colorized)

    seg_model.eval()
    pred_rgb_model.eval()
    pred_mask_model.eval()
    pred_colorized_mask_model.eval()

    # DATASET
    data_dir = os.path.join(cfg.data_dir, "test")
    test_data = SynpickVideoDataset(data_dir=data_dir, num_frames=VIDEO_TOT_LENGTH,
                                    step=4, allow_overlap=VID_DATA_ALLOW_OVERLAP)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    iter_loader = iter(test_loader)

    # METRICS
    def mse(pred, target):
        return np.square(np.subtract(pred, target)).mean()

    with torch.no_grad():
        for i in tqdm(range(1)):

            imgs, _, colorized_masks = next(iter_loader)
            imgs, colorized_masks = imgs.to(DEVICE), colorized_masks.to(DEVICE) # [1, T, 3, h, w]
            gt_rgb_vis = postprocess_img(imgs.squeeze(dim=0))  # [T, h, w, 3]
            gt_colorized_vis = postprocess_img(colorized_masks.squeeze(dim=0))  # [T, h, w, 3]
            input = imgs[:, :VIDEO_IN_LENGTH]  # [1, t, 3, h, w]

            pred_rgb = pred_rgb_model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)
            pred_rgb = torch.cat([input, pred_rgb], dim=1)  # [1, T, 3, h, w]
            pred_rgb_vis = postprocess_img(pred_rgb.squeeze(dim=0))  # [T, 3, h, w]

            pred_then_seg = torch.stack([seg_model(pred_rgb[:, i]) for i in range(pred_rgb.shape[1])], dim=1)
            pred_then_seg = pred_then_seg.argmax(dim=2).squeeze()  # [T, h, w]
            pred_seg_color_vis = colorize_semseg(postprocess_mask(pred_then_seg), num_classes=SYNPICK_CLASSES).transpose(0, 3, 1, 2) # [T, 3, h, w]

            seg = torch.stack([seg_model(imgs[:, i]) for i in range(imgs.shape[1])], dim=1).argmax(dim=2)  # [1, T, 1, h, w]
            seg_input = torch.stack([(seg == i) for i in range(SYNPICK_CLASSES)], dim=2).float()  # [1, T, c, h, w] one-hot float
            input_seg = seg_input[:, :VIDEO_IN_LENGTH]  # [1, t, c, h, w]
            seg_then_pred = pred_mask_model.pred_n(input_seg, pred_length=VIDEO_PRED_LENGTH).argmax(dim=2)  # [1, n, 1, h, w]
            seg_then_pred = torch.cat([input_seg.argmax(dim=2), seg_then_pred], dim=1).squeeze()  # [T, h, w]
            seg_pred_color_vis = colorize_semseg(postprocess_mask(seg_then_pred), num_classes=SYNPICK_CLASSES).transpose(0, 3, 1, 2)  # [T, 3, h, w]

            seg_colorized = colorize_semseg(postprocess_mask(seg.squeeze()), num_classes=SYNPICK_CLASSES)
            seg_color_per_frame_vis = seg_colorized.transpose(0, 3, 1, 2)  # [T, 3, h, w]

            input_colorized = preprocess_img(seg_colorized[:VIDEO_IN_LENGTH]).to(DEVICE).unsqueeze(dim=0)  # [b, t, 3, h, w]
            seg_color_pred = pred_colorized_mask_model.pred_n(input_colorized, pred_length=VIDEO_PRED_LENGTH)
            seg_color_pred = torch.cat([input_colorized, seg_color_pred], dim=1).squeeze(dim=0)
            seg_color_pred_vis = postprocess_img(seg_color_pred)  # [T, 3, h, w]

            print("")
            print(f"MSE loss seg->colorize (per frame): {mse(seg_color_per_frame_vis, gt_colorized_vis)}")
            print(f"MSE loss pred->seg->colorize: {mse(pred_seg_color_vis, gt_colorized_vis)}")
            print(f"MSE loss seg->pred->colorize: {mse(seg_pred_color_vis, gt_colorized_vis)}")
            print(f"MSE loss seg->colorize->pred: {mse(seg_color_pred_vis, gt_colorized_vis)}")
            print("")

            save_vid_vis(
                out_fp=os.path.join(cfg.out_dir, "4way_vis_{}.gif".format(str(i))),
                video_in_length=VIDEO_IN_LENGTH,
                True_Trajectory_RGB=gt_rgb_vis,
                True_Trajectory_Seg=gt_colorized_vis,
                Framewise_Segmentation=seg_color_per_frame_vis,
                RGB_Prediction=pred_rgb_vis,
                RGB_Prediction_Colorized=pred_seg_color_vis,
                Mask_Prediction_Colorized=seg_pred_color_vis,
                Colorization_Prediction=seg_color_pred_vis
            )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Video Prediction 4way vis")
    parser.add_argument("--seg", type=str, help="Path to segmentation model")
    parser.add_argument("--pred-rgb", type=str, help="Path to prediction model (rgb)")
    parser.add_argument("--pred-mask", type=str, help="Path to prediction model (masks)")
    parser.add_argument("--pred-colorized", type=str, help="Path to prediction model (colorized)")
    parser.add_argument("--data-dir", type=str, help="Path to data dir")
    parser.add_argument("--out-dir", type=str, help="Output path for results")

    cfg = parser.parse_args()
    visualize_4_way(cfg)