import sys, os

import numpy as np
import torch

from config import *
from dataset import MyDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
from utils import save_vis, colorize_semseg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SYNPICK_CLASSES = ['object_{}'.format(i) for i in range(1, 22)]


def visualize(data_dir, model_path):

    test_img_dir = os.path.join(data_dir, 'test', 'rgb')
    test_msk_dir = os.path.join(data_dir, 'test', 'masks')

    test_dataset = MyDataset(images_dir=test_img_dir, masks_dir=test_msk_dir,
                          classes=SYNPICK_CLASSES)

    seg_model = torch.load(path) if model_path is not None else None

    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image, gt_mask = test_dataset[n]
        image_vis = image.astype('uint8')
        gt_mask = gt_mask.squeeze()

        if seg_model is not None:
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())

            save_vis(
                out_fp="./vis{}.png".format(str(i)),
                image=image_vis,
                ground_truth_mask=colorize_semseg(gt_mask),
                predicted_mask=colorize_semseg(pr_mask)
            )

        else:
            save_vis(
                out_fp="./vis{}.png".format(str(i)),
                image=image_vis,
                ground_truth_mask=colorize_semseg(gt_mask)
            )


if __name__ == '__main__':
    visualize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)