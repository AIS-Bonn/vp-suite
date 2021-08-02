import sys

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SynpickVideoDataset, postprocess_mask, postprocess_img
from config import *
from utils import colorize_semseg, save_vid_vis

def visualize_4_way(cfg):

    # MODELS
    seg_model_path, pred_rgb_model_path, pred_mask_model_path, pred_colorized_mask_model_path = model_paths
    seg_model = torch.load(cfg.seg)
    pred_rgb_model = torch.load(cfg.pred_rgb)
    pred_mask_model = torch.load(cfg.pred_mask)
    pred_colorized_mask_model = torch.load(cfg.pred_colorized)

    # DATASET
    data_dir = os.path.join(cfg.data_dir, "test", "rgb")
    test_data = SynpickVideoDataset(data_dir=data_dir, vid_type=("rgb", 3), num_frames=VIDEO_TOT_LENGTH,
                                    step=4, allow_overlap=VID_DATA_ALLOW_OVERLAP)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)

    i = 0
    loop = tqdm(test_loader)
    for batch_idx, frames in enumerate(loop):

        if i >= 10: break

        frames = frames.to(DEVICE)  # [1, T, 3, h, w]
        frames_vis = postprocess_img(frames.squeeze(dim=0))  # [T, 3, h, w]
        input = frames[:, :VIDEO_IN_LENGTH]  # [1, t, 3, h, w]

        pred_rgb = pred_rgb_model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)  # [1, T, 3, h, w]
        pred_rgb_vis = postprocess_img(pred_rgb)  # [T, 3, h, w]

        pred_rgb = torch.cat([input, pred_rgb], dim=1)
        pred_rgb = torch.stack([seg_model(pred_rgb[:, i]) for i in range(pred_rgb.shape[1])], dim=1)
        pred_rgb = pred_rgb.argmax(dim=2).squeeze()  # [T, h, w]
        pred_then_colorized_vis = colorize_semseg(postprocess_mask(pred_rgb), num_classes=SYNPICK_CLASSES) # [T, 3, h, w]

        frames_seg = [seg_model(frames[:, i]).argmax(dim=1) for i in range(frames.shape[1])]
        frames_seg = torch.stack(frames_seg, dim=1)  # [1, 1, h, w]
        input_seg = frames_seg[:, :VIDEO_IN_LENGTH]  # [1, t, 1, h, w]

        pred_mask = pred_mask_model.pred_n(input_seg, pred_length=VIDEO_PRED_LENGTH)
        pred_mask = pred_mask.argmax(dim=2)  # [1, T, 1, h, w]
        pred_mask = postprocess_mask(torch.cat([input_seg, pred_mask], dim=1).squeeze())  # [T, h, w]
        pred_mask_vis = colorize_semseg(pred_mask, num_classes=SYNPICK_CLASSES)  # [T, 3, h, w]

        frames_colorized = colorize_semseg(postprocess_mask(frames_seg.squeeze()), num_classes=SYNPICK_CLASSES).unsqueeze(dim=0) # [1, T, 3, h, w]
        frames_colorized_vis = postprocess_img(frames_colorized.squeeze(dim=0))  # [T, 3, h, w]
        input_colorized = frames_colorized[:VIDEO_IN_LENGTH]

        colorized_then_pred = pred_colorized_mask_model.pred_n(input_colorized, pred_length=VIDEO_PRED_LENGTH)
        colorized_then_pred = torch.cat([input_colorized, colorized_then_pred], dim=1).squeeze(dim=0)
        colorized_then_pred_vis = postprocess_img(colorized_then_pred)  # [T, 3, h, w]

        save_vid_vis(
            out_fp=os.path.join(cfg.out_dir, "4way_vis_{}.gif".format(str(i))),
            video_in_length=VIDEO_IN_LENGTH,
            true_trajectory=frames_vis,
            gt_colorized=frames_colorized_vis,
            pr_rgb=pred_rgb_vis,
            pr_rgb_colorized=pred_then_colorized_vis,
            pr_mask=pred_mask_vis,
            pr_colorized=colorized_then_pred_vis
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