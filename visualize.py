import sys, os

import numpy as np
import torch

from config import *
from dataset import SynpickDataset, get_validation_augmentation
from utils import save_vis, colorize_semseg

def visualize(dataset, seg_model=None):

    for i in range(5):
        n = np.random.choice(len(dataset))

        image, gt_mask = dataset[n]
        print(image.shape, gt_mask.shape)
        image_vis = image.astype('uint8')

        if seg_model is not None:
            seg_model.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                print(x_tensor.shape)
                pr_mask = seg_model.predict(x_tensor)
                pr_mask = pr_mask.argmax(dim=1).squeeze().cpu().numpy().astype('uint8')

                save_vis(
                    out_fp="./vis{}.png".format(str(i)),
                    image=image_vis,
                    ground_truth_mask=colorize_semseg(gt_mask, num_classes=dataset.NUM_CLASSES),
                    predicted_mask=colorize_semseg(pr_mask, num_classes=dataset.NUM_CLASSES)
                )
            seg_model.train()

        else:
            save_vis(
                out_fp="./vis{}.png".format(str(i)),
                image=image_vis,
                ground_truth_mask=colorize_semseg(gt_mask, num_classes=dataset.NUM_CLASSES)
            )

if __name__ == '__main__':

    data_dir = sys.argv[1]
    test_img_dir = os.path.join(data_dir, 'test', 'rgb')
    test_msk_dir = os.path.join(data_dir, 'test', 'masks')
    test_dataset = SynpickDataset(images_dir=test_img_dir, masks_dir=test_msk_dir,
                                  augmentation=get_validation_augmentation(),
                                  classes=SYNPICK_CLASSES)
    if len(sys.argv) > 2:
        seg_model = torch.load(sys.argv[2])
        visualize(test_dataset, seg_model)
    visualize(test_dataset)