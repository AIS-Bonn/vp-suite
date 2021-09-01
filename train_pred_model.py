import os, time, argparse, random
from pathlib import Path

import numpy as np
import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from dataset import SynpickVideoDataset
from models.prediction.pred_model import CopyLastFrameModel, UNet3d, LSTMModel, ST_LSTM_NoEncode
from metrics.prediction.fvd import FrechetVideoDistance
from metrics.prediction.mse import MSE
from metrics.prediction.bce import BCELogits
from utils import validate_vid_model
from visualize import visualize_vid

def main(args):

    # INITIAL PREPPING
    best_val_loss = float("inf")
    timestamp = int(1000000 * time.time())
    out_dir = Path("out/{}_pred_model".format(timestamp))
    out_dir.mkdir(parents=True)

    num_classes = SYNPICK_CLASSES + 1 if cfg.include_gripper else SYNPICK_CLASSES
    num_channels = num_classes if cfg.pred_mode == "mask" else 3
    vid_type = (cfg.pred_mode, num_channels)

    # SEEDING
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.in_path
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_data = SynpickVideoDataset(data_dir=train_dir, num_frames=VIDEO_TOT_LENGTH,
                                     step=VID_STEP, allow_overlap=VID_DATA_ALLOW_OVERLAP, num_classes=num_classes)
    val_data = SynpickVideoDataset(data_dir=val_dir, num_frames=VIDEO_TOT_LENGTH,
                                   step=VID_STEP, allow_overlap=VID_DATA_ALLOW_OVERLAP, num_classes=num_classes)
    exit(0)
    train_loader = DataLoader(train_data, batch_size=VID_BATCH_SIZE, shuffle=True, num_workers=VID_BATCH_SIZE,
                              drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    # MODEL

    if cfg.model == "unet":
        print("prediction model: UNet3d")
        pred_model = UNet3d(in_channels=num_channels, out_channels=num_channels, time_dim=VIDEO_IN_LENGTH).to(DEVICE)
    elif cfg.model == "lstm":
        print("prediction model: LSTM")
        pred_model = LSTMModel(in_channels=num_channels, out_channels=num_channels).to(DEVICE)
    elif cfg.model == "st_lstm":
        print("prediction model: ST-LSTM")
        pred_model = ST_LSTM_NoEncode(img_size=train_data.img_shape, img_channels=num_channels, device=DEVICE)
    else:
        print("prediction model: CopyLastFrame")
        pred_model = CopyLastFrameModel().to(DEVICE)
        cfg.no_train = True

    # LOSSES
    # name: (loss_fn, use_full_input, scale)
    losses = {
        "mse": (MSE(), False, 1.0),
        "bce": (BCELogits(), False, 0.0)
    }
    # FVD loss only available for 2- or 3- channel input
    if num_channels == 2 or num_channels == 3:
        losses["fvd"] = (FrechetVideoDistance(num_frames=VIDEO_TOT_LENGTH, in_channels=num_channels), True, 0.00)
    # Check if indicator loss available
    if cfg.indicator_val_loss not in losses.keys():
        default_loss = "mse"
        print(f"Indicator loss {cfg.indicator_val_loss} only usable with 2 or 3 input channels -> using {default_loss}")
        cfg.indicator_val_loss = default_loss

    # OPTIMIZER
    optimizer = None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=LEARNING_RATE)

    # TRAINING
    for i in range(0, NUM_EPOCHS):
        print(f'\nEpoch: {i+1} of {NUM_EPOCHS}')

        if not cfg.no_train:
            print("Cur epoch training loss scales: {}"
                  .format(["{}: {}".format(name, scale) for name, (_, _, scale) in losses.items()]))
            loop = tqdm(train_loader)

            for batch_idx, (imgs, masks, colorized_masks) in enumerate(loop):

                if cfg.pred_mode == "rgb":
                    data = imgs
                elif cfg.pred_mode == "colorized":
                    data = colorized_masks
                else:
                    data = masks

                # fwd
                data = data.to(DEVICE)  # [b, T, c, h, w], with T = VIDEO_TOT_LENGTH

                input, targets = data[:, :VIDEO_IN_LENGTH], data[:, VIDEO_IN_LENGTH:]
                predictions, model_losses = pred_model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)

                # loss
                predictions_full = torch.cat([input, predictions], dim=1)
                targets_full = data
                loss = torch.tensor(0.0, device=DEVICE)
                for _, (loss_fn, use_full_input, scale) in losses.items():
                    if scale == 0: continue
                    pred = predictions_full if use_full_input else predictions
                    real = targets_full if use_full_input else targets
                    loss += scale * loss_fn(pred, real)
                if model_losses is not None:
                    for loss_value in model_losses.values():
                        loss += loss_value

                # bwd
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pred_model.parameters(), 10)
                optimizer.step()

                loop.set_postfix(loss=loss.item())
                # loop.set_postfix(mem=torch.cuda.memory_allocated())
        else:
            print("Skipping trianing loop.")

        print("Validating...")
        val_losses = validate_vid_model(valid_loader, pred_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, losses,
                                        cfg.pred_mode, num_channels)
        cur_val_loss = val_losses[cfg.indicator_val_loss]

        # save model if last epoch improved acc.
        if best_val_loss > cur_val_loss:
            best_val_loss = cur_val_loss
            torch.save(pred_model, str((out_dir / 'best_model.pth').resolve()))
            print(f"Minimum indicator loss ({cfg.indicator_val_loss}) reduced -> model saved!")

        # visualize model predictions using eval mode and validation data
        print("Saving visualizations...")
        visualize_vid(val_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, pred_model, out_dir, vid_type, num_vis=10)

        if i >= 30 and i % 10 == 0:
            optimizer.param_groups[0]['lr'] *= 0.5
            print('Decrease learning rate!')

    # TESTING
    print("\nTraining done, testing best model...")
    best_model = torch.load(str((out_dir / 'best_model.pth').resolve()))
    test_data = SynpickVideoDataset(data_dir=test_dir, num_frames=VIDEO_TOT_LENGTH, step=4, num_classes=num_channels)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    validate_vid_model(test_loader, best_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, losses,
                                        cfg.pred_mode, num_channels)

    print("Saving visualizations...")
    visualize_vid(test_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, best_model, out_dir, vid_type, num_vis=10)
    print("Testing done, bye bye!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Video Prediction Model Training")
    parser.add_argument("--no-train", action="store_true", help="If specified, the training loop is skipped")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--in-path", type=str, help="Path to dataset directory")
    parser.add_argument("--include-gripper", action="store_true", help="If specified, gripper is included in masks")
    parser.add_argument("--model", type=str, choices=["unet", "lstm", "st_lstm", "copy"], default="unet",
                        help="Which model arch to use")
    parser.add_argument("--pred-mode", type=str, choices=["rgb", "colorized", "mask"], default="rgb",
                        help="Which kind of data to train/test on")
    parser.add_argument("--indicator-val-loss", type=str, choices=["mse", "fvd", "bce"], default="fvd",
                        help="Loss to use for determining if validated model has become 'better' and should be saved")

    cfg = parser.parse_args()
    main(cfg)
