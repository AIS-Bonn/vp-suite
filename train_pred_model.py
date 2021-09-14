import os, time, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from dataset import SynpickVideoDataset
from models.prediction.pred_model_factory import get_pred_model
from metrics.prediction.fvd import FrechetVideoDistance
from metrics.prediction.mse import MSE
from metrics.prediction.bce import BCELogits
from visualize import visualize_vid

def main(args):

    # PREPARATION pt. 1
    best_val_loss = float("inf")
    timestamp = int(1000000 * time.time())
    out_dir = Path("out/{}_pred_model".format(timestamp))
    out_dir.mkdir(parents=True)

    num_classes = SYNPICK_CLASSES + 1 if cfg.include_gripper else SYNPICK_CLASSES
    num_channels = num_classes if cfg.pred_mode == "mask" else 3
    vid_type = (cfg.pred_mode, num_channels)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.in_path
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_data = SynpickVideoDataset(data_dir=train_dir, num_frames=VIDEO_TOT_LENGTH, step=VID_STEP,
                                     allow_overlap=VID_DATA_ALLOW_OVERLAP, num_classes=num_classes)
    val_data = SynpickVideoDataset(data_dir=val_dir, num_frames=VIDEO_TOT_LENGTH, step=VID_STEP,
                                   allow_overlap=VID_DATA_ALLOW_OVERLAP, num_classes=num_classes)
    train_loader = DataLoader(train_data, batch_size=VID_BATCH_SIZE, shuffle=True, num_workers=VID_BATCH_SIZE,
                              drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    cfg.action_size = train_data.action_size
    cfg.img_shape = train_data.img_shape

    # MODEL AND OPTIMIZER
    pred_model = get_pred_model(cfg, num_channels, VIDEO_IN_LENGTH, DEVICE)
    optimizer = None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=cfg.lr)
        optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.2, min_lr=1e-7)

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


    # MAIN LOOP
    for epoch in range(0, NUM_EPOCHS):

        # train
        print('\nTraining (epoch: {} of {}, loss scales: {})'
              .format(epoch+1, NUM_EPOCHS, ["{}: {}".format(name, scale) for name, (_, _, scale) in losses.items()]))
        if not cfg.no_train:

            # use prediction model's own training loop if available
            if callable(getattr(pred_model, "train_iter", None)):
                pred_model.train_iter(train_loader, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, cfg.pred_mode,
                                      optimizer, losses, epoch)
            else:
                train_iter(train_loader, pred_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, cfg.pred_mode,
                           optimizer, losses)
        else:
            print("Skipping trianing loop.")

        # eval.
        print("Validating...")
        val_losses = eval_iter(valid_loader, pred_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH,
                               cfg.pred_mode, losses)
        cur_val_loss = val_losses[cfg.indicator_val_loss]
        optimizer_scheduler.step(cur_val_loss)

        # save model if last epoch improved acc.
        if best_val_loss > cur_val_loss:
            best_val_loss = cur_val_loss
            torch.save(pred_model, str((out_dir / 'best_model.pth').resolve()))
            print(f"Minimum indicator loss ({cfg.indicator_val_loss}) reduced -> model saved!")

        # visualize current model performance every nth epoch, using eval mode and validation data.
        if epoch % 10 == 9:
            print("Saving visualizations...")
            visualize_vid(val_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, pred_model, out_dir, vid_type, num_vis=10)

    # TESTING
    print("\nTraining done, testing best model...")
    best_model = torch.load(str((out_dir / 'best_model.pth').resolve()))
    test_data = SynpickVideoDataset(data_dir=test_dir, num_frames=VIDEO_TOT_LENGTH, step=4,
                                    allow_overlap=VID_DATA_ALLOW_OVERLAP, num_classes=num_classes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    eval_iter(test_loader, best_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, cfg.pred_mode, losses)

    print("Saving visualizations...")
    visualize_vid(test_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, best_model, out_dir, vid_type, num_vis=10)
    print("Testing done, bye bye!")



def train_iter(loader, pred_model, device, video_in_length, video_pred_length, pred_mode, optimizer, losses):

    loop = tqdm(loader)
    for batch_idx, data in enumerate(loop):

        # fwd
        img_data = data[cfg.pred_mode].to(device)  # [b, T, c, h, w], with T = VIDEO_TOT_LENGTH
        input, targets = img_data[:, :video_in_length], img_data[:, video_in_length:]
        actions = data["actions"].to(device)

        predictions, model_losses = pred_model.pred_n(input, pred_length=video_pred_length, actions=actions)

        # loss
        predictions_full = torch.cat([input, predictions], dim=1)
        targets_full = img_data
        loss = torch.tensor(0.0, device=device)
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
        torch.nn.utils.clip_grad_norm_(pred_model.parameters(), 100)
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        # loop.set_postfix(mem=torch.cuda.memory_allocated())


def eval_iter(loader, pred_model, device, video_in_length, video_pred_length, pred_mode, losses):

    pred_model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        all_losses = {key: [] for key in losses.keys()}
        for batch_idx, data in enumerate(loop):

            # fwd
            img_data = data[pred_mode].to(device)  # [b, T, h, w], with T = video_tot_length
            input, targets = img_data[:, :video_in_length], img_data[:, video_in_length:]
            actions = data["actions"].to(device)

            predictions, model_losses = pred_model.pred_n(input, pred_length=video_pred_length, actions=actions)

            # metrics
            predictions_full = torch.cat([input, predictions], dim=1)
            targets_full = img_data
            for name, (loss_fn, use_full_input, _) in losses.items():
                pred = predictions_full if use_full_input else predictions
                real = targets_full if use_full_input else targets
                loss = loss_fn(pred, real).item()
                all_losses[name].append(loss)
            if model_losses is not None:
                for loss_name, loss_value in model_losses.items():
                    if loss_name in all_losses.keys():
                        all_losses[loss_name].append(loss_value.item())
                    else:
                        all_losses[loss_name] = [loss_value.item()]

    pred_model.train()

    print("Validation losses:")
    for key in all_losses.keys():
        cur_losses = all_losses[key]
        avg_loss = sum(cur_losses) / len(cur_losses)
        print(f" - {key}: {avg_loss}")
        all_losses[key] = avg_loss

    return all_losses

# ==============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Video Prediction Model Training")
    parser.add_argument("--no-train", action="store_true", help="If specified, the training loop is skipped")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--in-path", type=str, help="Path to dataset directory")
    parser.add_argument("--include-gripper", action="store_true", help="If specified, gripper is included in masks")
    parser.add_argument("--include-actions", action="store_true", help="use gripper deltas for action-conditional learning")
    parser.add_argument("--model", type=str, choices=["unet", "lstm", "st_lstm", "copy", "phy", "st_phy"],
                        default="st_lstm", help="Which model arch to use")
    parser.add_argument("--pred-mode", type=str, choices=["rgb", "colorized", "mask"], default="rgb",
                        help="Which kind of data to train/test on")
    parser.add_argument("--indicator-val-loss", type=str, choices=["mse", "fvd", "bce"], default="fvd",
                        help="Loss to use for determining if validated model has become 'better' and should be saved")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    cfg = parser.parse_args()
    main(cfg)


