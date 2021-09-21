import os, time, random
from pathlib import Path

import wandb

import numpy as np
import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from scripts.compare_pred_models import test_pred_models
from dataset import SynpickVideoDataset
from models.prediction.pred_model_factory import get_pred_model
from losses.main import PredictionLossProvider
from visualize import visualize_vid

def train(cfg):

    # PREPARATION pt. 1
    best_val_loss = float("inf")
    best_model_path = str((Path(cfg.out_dir) / 'best_model.pth').resolve())
    num_classes = cfg.dataset_classes + 1 if cfg.include_gripper else cfg.dataset_classes
    cfg.num_channels = num_classes if cfg.pred_mode == "mask" else 3
    vid_type = (cfg.pred_mode, cfg.num_channels)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_data = SynpickVideoDataset(data_dir=train_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap, num_classes=num_classes,
                                     include_gripper=cfg.include_gripper)
    val_data = SynpickVideoDataset(data_dir=val_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                   allow_overlap=cfg.vid_allow_overlap, num_classes=num_classes,
                                   include_gripper=cfg.include_gripper)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.batch_size,
                              drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    cfg.action_size = train_data.action_size
    cfg.img_shape = train_data.img_shape

    # WandB
    wandb.init(project="sem_vp_train_pred", config=cfg)
    cfg = wandb.config

    # MODEL AND OPTIMIZER
    pred_model = get_pred_model(cfg)
    optimizer = None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=cfg.lr)
        optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                         min_lr=1e-6, verbose=True)

    # LOSSES
    loss_provider = PredictionLossProvider(cfg)
    # Check if indicator loss available
    if loss_provider.losses.get(cfg.pred_val_criterion, (None, 0.0))[1] <= 0.0:
        cfg.pred_val_criterion = "mse"

    # MAIN LOOP
    for epoch in range(0, cfg.epochs):

        # train
        print(f'\nTraining (epoch: {epoch+1} of {cfg.epochs})')
        if not cfg.no_train:
            # use prediction model's own training loop if available
            if callable(getattr(pred_model, "train_iter", None)):
                pred_model.train_iter(cfg, train_loader, optimizer, loss_provider, epoch)
            else:
                train_iter(cfg, train_loader, pred_model, optimizer, loss_provider)
        else:
            print("Skipping trianing loop.")

        # eval
        print("Validating...")
        val_losses, indicator_loss = eval_iter(cfg, valid_loader, pred_model, loss_provider)
        optimizer_scheduler.step(indicator_loss)
        print("Validation losses (mean over entire validation set):")
        for k, v in val_losses.items():
            print(f" - {k}: {v}")

        # save model if last epoch improved indicator loss
        cur_val_loss = indicator_loss.item()
        if best_val_loss > cur_val_loss:
            best_val_loss = cur_val_loss
            torch.save(pred_model, best_model_path)
            print(f"Minimum indicator loss ({cfg.pred_val_criterion}) reduced -> model saved!")

        # visualize current model performance every nth epoch, using eval mode and validation data.
        if epoch % 10 == 9:
            print("Saving visualizations...")
            out_filenames = visualize_vid(val_data, cfg.vid_input_length, cfg.vid_pred_length, pred_model,
                                          cfg.device, cfg.out_dir, vid_type, num_vis=10)

            log_vids = {f"vis_{i}": wandb.Video(out_fn, fps=4,format="gif") for i, out_fn in enumerate(out_filenames)}
            wandb.log(log_vids, commit=False)

        # final bookkeeping
        wandb.log({"sweep_loss": val_losses["mse"]}, commit=False)
        wandb.log(val_losses, commit=True)

    # TESTING
    print("\nTraining done, testing best model...")
    cfg.models = [best_model_path]
    cfg.full_evaluation = True
    test_pred_models(cfg)

    print("Testing done, bye bye!")

# ==============================================================================

def train_iter(cfg, loader, pred_model, optimizer, loss_provider):

    loop = tqdm(loader)
    for batch_idx, data in enumerate(loop):

        # input
        img_data = data[cfg.pred_mode].to(cfg.device)  # [b, T, c, h, w], with T = cfg.vid_total_length
        input, targets = img_data[:, :cfg.vid_input_length], img_data[:, cfg.vid_input_length:cfg.vid_total_length]
        actions = data["actions"].to(cfg.device)  # [b, T-1, a]. Action t corresponds to what happens after frame t

        # fwd
        predictions, model_losses = pred_model.pred_n(input, pred_length=cfg.vid_pred_length, actions=actions)

        # loss
        _, total_loss = loss_provider.get_losses(predictions, targets)
        if model_losses is not None:
            for value in model_losses.values():
                total_loss += value

        # bwd
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(pred_model.parameters(), 100)
        optimizer.step()

        # bookkeeping
        loop.set_postfix(loss=total_loss.item())
        # loop.set_postfix(mem=torch.cuda.memory_allocated())


def eval_iter(cfg, loader, pred_model, loss_provider):

    pred_model.eval()
    loop = tqdm(loader)
    all_losses = []
    indicator_losses = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loop):

            # fwd
            img_data = data[cfg.pred_mode].to(cfg.device)  # [b, T, h, w], with T = vid_total_length
            input, targets = img_data[:, :cfg.vid_input_length], img_data[:, cfg.vid_input_length:cfg.vid_total_length]
            actions = data["actions"].to(cfg.device)

            predictions, model_losses = pred_model.pred_n(input, pred_length=cfg.vid_pred_length, actions=actions)

            # metrics
            loss_values, _ = loss_provider.get_losses(predictions, targets, eval=True)
            if model_losses is not None:
                for k, v in model_losses.items():
                    loss_values[k] = v
            all_losses.append(loss_values)
            indicator_losses.append(loss_values[cfg.pred_val_criterion])

    indicator_loss = torch.stack(indicator_losses).mean()
    all_losses = {
        k: torch.stack([loss_values[k] for loss_values in all_losses]).mean().item() for k in all_losses[0].keys()
    }
    pred_model.train()

    return all_losses, indicator_loss