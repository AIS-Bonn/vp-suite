import sys, os

import numpy as np
import torch

from config import *
from dataset import SynpickSegmentationDataset, postprocess_img, postprocess_mask
from utils import save_vis, save_video_vis, colorize_semseg, synpick_seg_val_augmentation


def visualize(dataset, seg_model=None, out_dir=".", num_vis=5):

    for i in range(num_vis):
        n = np.random.choice(len(dataset))

        image, gt_mask = dataset[n]
        image_vis = postprocess_img(image.permute((1, 2, 0)))
        gt_mask_vis = postprocess_mask(gt_mask.squeeze())

        if seg_model is not None:
            seg_model.eval()
            with torch.no_grad():
                pr_mask = seg_model(image.to(DEVICE).unsqueeze(0))
                pr_mask_vis = postprocess_mask(pr_mask.argmax(dim=1).squeeze())

                save_vis(
                    out_fp=os.path.join(out_dir, "{}.png".format(str(i))),
                    image=image_vis,
                    ground_truth_mask=colorize_semseg(gt_mask_vis, num_classes=dataset.NUM_CLASSES),
                    predicted_mask=colorize_semseg(pr_mask_vis, num_classes=dataset.NUM_CLASSES)
                )
            seg_model.train()

        else:
            save_vis(
                out_fp=os.path.join(out_dir, "{}.png".format(str(i))),
                image=image_vis,
                ground_truth_mask=colorize_semseg(gt_mask_vis, num_classes=dataset.NUM_CLASSES)
            )


def visualize_video(dataset, video_in_length, video_pred_length, pred_model=None, out_dir=".", num_vis=5):

    for i in range(num_vis):
        n = np.random.choice(len(dataset))

        gt_traj = dataset[n] # [in_l + pred_l, c, h, w]
        gt_traj_vis = postprocess_img(gt_traj)

        if pred_model is not None:
            pred_model.eval()
            with torch.no_grad():
                in_traj = gt_traj[:video_in_length].to(DEVICE).unsqueeze(dim=0)  # [1, in_l, c, h, w]
                pr_traj = pred_model.pred_n(in_traj, video_pred_length)  # [1, pred_l, c, h, w]
                pr_traj = torch.cat([in_traj, pr_traj], dim=1)  # [1, in_l + pred_l, c, h, w]
                pr_traj_vis = postprocess_img(pr_traj.squeeze(dim=0))  # [in_l + pred_l, c, h, w]

                save_video_vis(
                    out_fp=os.path.join(out_dir, "{}.gif".format(str(i))),
                    video_in_length=video_in_length,
                    true_trajectory=gt_traj_vis,
                    pred_trajectory=pr_traj_vis
                )

            pred_model.train()

        else:
            save_video_vis(
                out_fp=os.path.join(out_dir, "{}.gif".format(str(i))),
                video_in_length=video_in_length,
                true_trajectory=gt_traj_vis,
            )


if __name__ == '__main__':

    data_dir = sys.argv[1]
    test_img_dir = os.path.join(data_dir, 'test', 'rgb')
    test_msk_dir = os.path.join(data_dir, 'test', 'masks')
    test_dataset = SynpickSegmentationDataset(data_dir=os.path.join(data_dir, 'test'), augmentation=synpick_seg_val_augmentation())
    if len(sys.argv) > 2:
        seg_model = torch.load(sys.argv[2])
        visualize(test_dataset, seg_model)
    visualize(test_dataset)