import sys, os, time, random, argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import SynpickSegmentationDataset, synpick_seg_val_augmentation, synpick_seg_train_augmentation
from models.segmentation.seg_model import UNet
from utils import validate_seg_model
from metrics.segmentation.ce import CrossEntropyLoss
from visualize import visualize_seg

def main(cfg):

    # SEEDING
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.in_path
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_data = SynpickSegmentationDataset(data_dir=train_dir, augmentation=synpick_seg_train_augmentation())
    val_data = SynpickSegmentationDataset(data_dir=val_dir, augmentation=synpick_seg_val_augmentation())

    train_loader = DataLoader(train_data, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    # MODEL
    seg_model = UNet(in_channels=3, out_channels=train_data.NUM_CLASSES).to(DEVICE)

    # ETC
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=seg_model.parameters(), lr=LEARNING_RATE)

    max_accuracy = 0
    timestamp = int(1000000 * time.time())
    out_dir = Path("out/{}_seg_model".format(timestamp))
    out_dir.mkdir(parents=True)

    # TRAINING
    for i in range(0, NUM_EPOCHS):
        print('\nEpoch: {}'.format(i))

        loop = tqdm(train_loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = seg_model(data)
            loss = loss_fn.get_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        print("Validating...")
        accuracy = validate_seg_model(valid_loader, seg_model, DEVICE)
        print("Accuracy = {}".format(accuracy))

        # save model if last epoch improved acc.
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
    test_data = SynpickSegmentationDataset(data_dir=test_dir, augmentation=synpick_seg_val_augmentation())
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    accuracy = validate_seg_model(test_loader, best_model, DEVICE)
    print("Accuracy = {}".format(accuracy))

    visualize_seg(test_data, best_model, out_dir)
    print("Testing done, bye bye!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Semantic Segmentation Model Training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--in-path", type=str, help="Path to dataset directory")

    cfg = parser.parse_args()
    main(cfg)