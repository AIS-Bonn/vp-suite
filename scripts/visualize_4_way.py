import sys, os, argparse
sys.path.append(".")

from tqdm import tqdm

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
    data_dir = os.path.join(cfg.data_dir, "test", "rgb")
    test_data = SynpickVideoDataset(data_dir=data_dir, vid_type=("rgb", 3), num_frames=VIDEO_TOT_LENGTH,
                                    step=4, allow_overlap=VID_DATA_ALLOW_OVERLAP)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    iter_loader = iter(test_loader)

    with torch.no_grad():
        for i in tqdm(range(10)):

            frames = next(iter_loader).to(DEVICE)  # [1, T, 3, h, w]
            frames_vis = postprocess_img(frames.squeeze(dim=0))  # [T, 3, h, w]
            input = frames[:, :VIDEO_IN_LENGTH]  # [1, t, 3, h, w]

            pred_rgb = pred_rgb_model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)
            pred_rgb = torch.cat([input, pred_rgb], dim=1)  # [1, T, 3, h, w]
            pred_rgb_vis = postprocess_img(pred_rgb.squeeze(dim=0))  # [T, 3, h, w]

            pred_rgb = torch.stack([seg_model(pred_rgb[:, i]) for i in range(pred_rgb.shape[1])], dim=1)
            pred_rgb = pred_rgb.argmax(dim=2).squeeze()  # [T, h, w]
            pred_then_colorized_vis = colorize_semseg(postprocess_mask(pred_rgb), num_classes=SYNPICK_CLASSES).transpose(0, 3, 1, 2) # [T, 3, h, w]

            frames_seg = torch.stack([seg_model(frames[:, i]) for i in range(frames.shape[1])], dim=1).argmax(dim=2)  # [1, T, 1, h, w]
            frames_seg_in = torch.stack([(frames_seg == i) for i in range(SYNPICK_CLASSES)], dim=2).float()  # [1, T, c, h, w] one-hot float
            input_seg = frames_seg_in[:, :VIDEO_IN_LENGTH]  # [1, t, c, h, w]
            pred_mask = pred_mask_model.pred_n(input_seg, pred_length=VIDEO_PRED_LENGTH).argmax(dim=2)  # [1, n, 1, h, w]
            pred_mask = torch.cat([input_seg.argmax(dim=2), pred_mask], dim=1).squeeze()  # [T, h, w]
            pred_mask_vis = colorize_semseg(postprocess_mask(pred_mask), num_classes=SYNPICK_CLASSES).transpose(0, 3, 1, 2)  # [T, 3, h, w]

            frames_colorized = colorize_semseg(postprocess_mask(frames_seg.squeeze()), num_classes=SYNPICK_CLASSES)
            frames_colorized_vis = frames_colorized.transpose(0, 3, 1, 2)  # [T, 3, h, w]

            input_colorized = preprocess_img(frames_colorized[:VIDEO_IN_LENGTH]).to(DEVICE).unsqueeze(dim=0)  # [b, t, 3, h, w]
            colorized_then_pred = pred_colorized_mask_model.pred_n(input_colorized, pred_length=VIDEO_PRED_LENGTH)
            colorized_then_pred = torch.cat([input_colorized, colorized_then_pred], dim=1).squeeze(dim=0)
            colorized_then_pred_vis = postprocess_img(colorized_then_pred)  # [T, 3, h, w]

            save_vid_vis(
                out_fp=os.path.join(cfg.out_dir, "4way_vis_{}.gif".format(str(i))),
                video_in_length=VIDEO_IN_LENGTH,
                True_Trajectory=frames_vis,
                Framewise_Segmentation=frames_colorized_vis,
                RGB_Prediction=pred_rgb_vis,
                RGB_Prediction_Colorized=pred_then_colorized_vis,
                Mask_Prediction_Colorized=pred_mask_vis,
                Colorization_Prediction=colorized_then_pred_vis
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