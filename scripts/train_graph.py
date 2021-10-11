import os, random
from pathlib import Path

import wandb

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.graph.synpick_graph import SynpickGraphDataset
from dataset.graph.pygt_loader import DataLoader
from models.graph_pred.graph_model_factory import get_graph_model
from utils.visualization import visualize_graph


def train(trial=None, cfg=None):

    # PREPARATION pt. 1
    best_eval_loss = float("inf")
    best_model_path = str((Path(cfg.out_dir) / 'best_model.pth').resolve())
    cfg.graph_out_size = 3 if cfg.graph_mode == "t" else 8

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_data = SynpickGraphDataset(data_dir=train_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap, graph_mode=cfg.graph_mode)
    val_data = SynpickGraphDataset(data_dir=val_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap, graph_mode=cfg.graph_mode)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=min(cfg.batch_size, 32),
                              drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    # WandB
    if not cfg.no_wandb:
        wandb.init(config=cfg, project="sem_vp_train_graph", reinit=False)

    # MODEL AND OPTIMIZER
    pred_model = get_graph_model(cfg, in_features=9).to(cfg.device)
    optimizer = None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=cfg.lr)
        optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                         min_lr=1e-6, verbose=True)

    # MAIN LOOP
    for epoch in range(0, cfg.epochs):

        # train
        print(f'\nTraining (epoch: {epoch+1} of {cfg.epochs})')
        train_iter(cfg, train_loader, pred_model, optimizer)

        # eval
        print("Validating...")
        vis_idx = [-1]
        if (epoch + 1) % cfg.vis_every_n == 0 and not cfg.no_vis:
            vis_idx = sorted(random.sample(range(len(valid_loader)), 10)) + vis_idx
        eval_loss, vis_pairs = eval_iter(cfg, valid_loader, pred_model, vis_idx)
        optimizer_scheduler.step(eval_loss)
        eval_loss = eval_loss.item()
        print(f"Validation loss (mean over entire validation set): {eval_loss}")

        # save model if last epoch improved indicator loss
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
            torch.save(pred_model, best_model_path)
            print(f"Minimum indicator loss reduced -> model saved!")

        # visualize current model performance every nth epoch, using eval mode and validation data.
        if vis_pairs != []:
            print("Saving visualizations...")
            out_filenames = visualize_graph(cfg, vis_pairs)

            if not cfg.no_wandb:
                log_vids = {f"vis_{i}": wandb.Video(out_fn, fps=3,format="gif")
                            for i, out_fn in enumerate(out_filenames)}
                wandb.log(log_vids, commit=False)

        # final bookkeeping
        if not cfg.no_wandb:
            wandb.log({"loss": eval_loss}, commit=True)

    # TESTING
    print("\nTraining done, testing best model...")
    cfg.models = [best_model_path]
    cfg.full_evaluation = True
    test_data = SynpickGraphDataset(data_dir=test_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap, graph_mode=cfg.graph_mode)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    vis_idx = sorted(random.sample(range(len(valid_loader)), 10)) + [-1]
    test_loss, _ = eval_iter(cfg, test_loader, pred_model, vis_idx)
    test_loss = test_loss.item()

    print(f"Test loss: {test_loss}")
    if not cfg.no_wandb:
        wandb.log({"test_loss": test_loss}, commit=True)
        wandb.finish()

    print("Testing done, bye bye!")
    return test_loss

# ==============================================================================

def train_iter(cfg, loader, pred_model, optimizer):

    loop = tqdm(loader)
    for _, signal_in in enumerate(loop):

        _, loss = pred_model.pred_n(signal_in, cfg.device, cfg.vid_pred_length)

        # bwd
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pred_model.parameters(), 100)
        optimizer.step()

        # bookkeeping
        loop.set_postfix(loss=loss.item())
        #loop.set_postfix(mem=torch.cuda.memory_allocated())


def eval_iter(cfg, loader, pred_model, vis_idx):

    pred_model.eval()
    loop = tqdm(loader)
    eval_losses = []

    next_vis_idx = vis_idx.pop(0)
    vis_pairs = []

    with torch.no_grad():

        for idx, signal_in in enumerate(loop):

            signal_pred, loss = pred_model.pred_n(signal_in, cfg.device, cfg.vid_pred_length)

            eval_losses.append(loss)
            loop.set_postfix(loss=loss.item())
            #loop.set_postfix(mem=torch.cuda.memory_allocated())

            if idx == next_vis_idx:
                vis_pairs.append((signal_pred, signal_in))
                next_vis_idx = vis_idx.pop(0)

    eval_loss = torch.stack(eval_losses).mean()
    pred_model.train()

    return eval_loss, vis_pairs