import sys, os

import numpy as np
import torch

from config import *
from dataset import SynpickSegmentationDataset
from utils import save_vis, colorize_semseg, synpick_seg_val_augmentation


def visualize(dataset, seg_model=None, out_dir="."):

    for i in range(5):
        n = np.random.choice(len(dataset))

        image, gt_mask = dataset[n]
        image_vis = image.permute((1, 2, 0)).cpu().numpy().astype('uint8')
        gt_mask_vis = gt_mask.squeeze().cpu().numpy().astype('uint8')

        if seg_model is not None:
            seg_model.eval()
            with torch.no_grad():
                pr_mask = seg_model(image.to(DEVICE).unsqueeze(0))
                pr_mask_vis = pr_mask.argmax(dim=1).squeeze().cpu().numpy().astype('uint8')

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

if __name__ == '__main__':

    data_dir = sys.argv[1]
    test_img_dir = os.path.join(data_dir, 'test', 'rgb')
    test_msk_dir = os.path.join(data_dir, 'test', 'masks')
    test_dataset = SynpickSegmentationDataset(data_dir=os.path.join(data_dir, 'test'), augmentation=synpick_seg_val_augmentation())
    if len(sys.argv) > 2:
        seg_model = torch.load(sys.argv[2])
        visualize(test_dataset, seg_model)
    visualize(test_dataset)