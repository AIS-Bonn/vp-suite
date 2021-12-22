import json
import random
from pathlib import Path

import wandb

import numpy as np
import torch.nn
from tqdm import tqdm

from dataset.factory import create_train_val_dataset
from vp_suite.models.factory import create_pred_model
from vp_suite.utils.img_processor import ImgProcessor
from evaluation.loss_provider import PredictionLossProvider
from utils.visualization import visualize_vid

def train(trial=None, cfg=None):

    # PREPARATION pt. 1
    best_val_loss = float("inf")
    best_model_path = str((Path(cfg.out_dir) / 'best_model.pth').resolve())
    cfg.img_processor = ImgProcessor(cfg.tensor_value_range)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.opt_direction == "maximize":
        def loss_improved(cur_loss, best_loss): return cur_loss > best_loss
    else:
        def loss_improved(cur_loss, best_loss): return cur_loss < best_loss

    # DATA
    (train_data, val_data), (train_loader, val_loader) = create_train_val_dataset(cfg)

    # Optuna
    if cfg.use_optuna:
        cfg.lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        cfg.mse_loss_scale = trial.suggest_float("mse_loss_scale", 1e-7, 1.0)
        cfg.l1_loss_scale = trial.suggest_float("l1_loss_scale", 1e-7, 1.0)
        cfg.smoothl1_loss_scale = trial.suggest_float("smoothl1_loss_scale", 1e-7, 1.0)
        cfg.lpips_loss_scale = trial.suggest_float("lpips_loss_scale", 1e-7, 1000.0)
        cfg.fvd_loss_scale = trial.suggest_float("fvd_loss_scale", 1e-7, 1.0)
        cfg.pred_st_decouple_loss_scale = trial.suggest_float("pred_st_decouple_loss_scale", 1e-7, 10000.0, log=True)
        cfg.pred_st_rec_loss_scale = trial.suggest_float("pred_st_rec_loss_scale", 1e-7, 1.0, log=True)
        cfg.pred_phy_moment_loss_scale = trial.suggest_float("pred_phy_moment_loss_scale", 1e-7, 10.0, log=True)

    # WandB
    if not cfg.no_wandb:
        wandb.init(config=cfg, project="sem_vp_train_pred", reinit=cfg.use_optuna)

    # MODEL AND OPTIMIZER
    pred_model = create_pred_model(cfg)
    optimizer, optimizer_scheduler = None, None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=cfg.lr)
        optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                         min_lr=1e-6, verbose=True)

    # LOSSES
    loss_provider = PredictionLossProvider(cfg)
    # Check if indicator loss available
    if loss_provider.losses.get(cfg.pred_val_criterion, (None, 0.0))[1] <= 1e-7:
        cfg.pred_val_criterion = "mse"

    # PREPARATION pt.2
    with open(str((Path(cfg.out_dir) / 'run_cfg.json').resolve()), "w") as cfg_file:
        json.dump(vars(cfg), cfg_file)

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
        val_losses, indicator_loss = eval_iter(cfg, val_loader, pred_model, loss_provider)
        if not cfg.no_train:
            optimizer_scheduler.step(indicator_loss)
        print("Validation losses (mean over entire validation set):")
        for k, v in val_losses.items():
            print(f" - {k}: {v}")

        # save model if last epoch improved indicator loss
        cur_val_loss = indicator_loss.item()
        if loss_improved(cur_val_loss, best_val_loss):
            best_val_loss = cur_val_loss
            torch.save(pred_model, best_model_path)
            print(f"Minimum indicator loss ({cfg.pred_val_criterion}) reduced -> model saved!")

        # visualize current model performance every nth epoch, using eval mode and validation data.
        if (epoch+1) % cfg.vis_every == 0 and not cfg.no_vis:
            print("Saving visualizations...")
            out_filenames = visualize_vid(val_data, cfg.context_frames, cfg.pred_frames, pred_model,
                                          cfg.device, cfg.img_processor, cfg.out_dir, num_vis=10)

            if not cfg.no_wandb:
                log_vids = {f"vis_{i}": wandb.Video(out_fn, fps=4,format="gif") for i, out_fn in enumerate(out_filenames)}
                wandb.log(log_vids, commit=False)

        # final bookkeeping
        if not cfg.no_wandb:
            wandb.log(val_losses, commit=True)

    # finishing
    print("\nTraining done, cleaning up...")
    torch.save(pred_model, str((Path(cfg.out_dir) / 'final_model.pth').resolve()))
    wandb.finish()
    return best_val_loss  # return best validation loss for hyperparameter optimization

# ==============================================================================

def train_iter(cfg, loader, pred_model, optimizer, loss_provider):

    loop = tqdm(loader)
    for batch_idx, data in enumerate(loop):

        # input
        img_data = data[cfg.pred_mode].to(cfg.device)  # [b, T, c, h, w], with T = cfg.vid_total_length
        input, targets = img_data[:, :cfg.context_frames], img_data[:, cfg.context_frames:cfg.vid_total_length]
        actions = data["actions"].to(cfg.device)  # [b, T-1, a]. Action t corresponds to what happens after frame t

        # fwd
        predictions, model_losses = pred_model.pred_n(input, pred_length=cfg.pred_frames, actions=actions)

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
            input, targets = img_data[:, :cfg.context_frames], img_data[:, cfg.context_frames:cfg.vid_total_length]
            actions = data["actions"].to(cfg.device)

            predictions, model_losses = pred_model.pred_n(input, pred_length=cfg.pred_frames, actions=actions)

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