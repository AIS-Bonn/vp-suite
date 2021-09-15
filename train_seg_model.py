import sys, os, time, random, argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn
from torch.utils.data import DataLoader

from config import *
from dataset import SynpickSegmentationDataset, synpick_seg_val_augmentation, synpick_seg_train_augmentation
from models.segmentation.seg_model import UNet
from visualize import visualize_seg

def main(cfg):

    # PREPARATION pt. 1
    num_classes = SYNPICK_CLASSES + 1 if cfg.include_gripper else SYNPICK_CLASSES
    device = DEVICE
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.in_path
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_data = SynpickSegmentationDataset(data_dir=train_dir, num_classes=num_classes)
    train_data.augmentation = synpick_seg_train_augmentation(img_h=train_data.img_h)
    val_data = SynpickSegmentationDataset(data_dir=val_dir, num_classes=num_classes)
    val_data.augmentation = synpick_seg_val_augmentation(img_h=val_data.img_h)

    train_loader = DataLoader(train_data, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    # MODEL, LOSSES, OPTIMIZERS
    seg_model = UNet(in_channels=3, out_channels=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=seg_model.parameters(), lr=LEARNING_RATE)

    # PREPARATION pt. 2
    max_accuracy = 0
    timestamp = int(1000000 * time.time())
    out_dir = Path("out/{}_seg_model".format(timestamp))
    out_dir.mkdir(parents=True)

    # MAIN LOOP
    for i in range(0, NUM_EPOCHS):

        train_iter(train_loader, seg_model, optimizer, loss_fn)
        accuracy = eval_iter(valid_loader, seg_model, device)

        # save model if last epoch improved acc.
        print("Accuracy = {}".format(accuracy))
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            torch.save(seg_model, str((out_dir / 'best_model.pth').resolve()))
            print('Model saved!')

        # visualize model predictions using eval mode and validation data
        visualize_seg(val_data, seg_model, out_dir)

        if i == 25:
            optimizer.param_groups[0]['lr'] *= 0.1
            print('Decrease decoder learning rate!')

    # TESTING
    print("\nTraining done, testing best model...")
    best_model = torch.load(str((out_dir / 'best_model.pth').resolve()))
    test_data = SynpickSegmentationDataset(data_dir=test_dir, num_classes=num_classes)
    test_data.augmentation = synpick_seg_val_augmentation(img_h=test_data.img_h)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    accuracy = eval_iter(test_loader, best_model, device)
    print("Accuracy = {}".format(accuracy))

    visualize_seg(test_data, best_model, out_dir)
    print("Testing done, bye bye!")


def train_iter(loader, seg_model, optimizer, loss_fn, device):

    print('\nEpoch: {}'.format(i))
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        predictions = seg_model(data)
        pred_ = predictions.permute((0, 2, 3, 1)).reshape(-1, predictions.shape[1])
        targets_ = targets.view(-1).long()
        loss = loss_fn(pred_, targets_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def eval_iter(loader, seg_model, device):

    print("Validating...")
    num_correct = 0
    num_pixels = 0
    seg_model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)  # shapes: [1, 3, h, w] for x and [1, h, w] for y
            preds = torch.argmax(seg_model(data), dim=1)   # [1, h, w]
            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)

    seg_model.train()

    return 100.0 * num_correct / num_pixels


# ==============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Semantic Segmentation Model Training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--in-path", type=str, help="Path to dataset directory")
    parser.add_argument("--include-gripper", action="store_true", help="If specified, gripper is included in masks")

    cfg = parser.parse_args()
    main(cfg)