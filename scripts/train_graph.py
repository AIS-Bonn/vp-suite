import os, time, random
from pathlib import Path

import wandb

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.graph.synpick_graph import SynpickGraphDataset, draw_synpick_graph, draw_synpick_pred_and_gt
from dataset.graph.pygt_loader import DataLoader
from torch_geometric.data import Data as GraphData
from models.graph_pred.rgcn import RecurrentGCN

def train(trial=None, cfg=None):

    # PREPARATION pt. 1
    best_eval_loss = float("inf")
    best_model_path = str((Path(cfg.out_dir) / 'best_model.pth').resolve())

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # DATA
    data_dir = cfg.data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_data = SynpickGraphDataset(data_dir=train_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap)
    val_data = SynpickGraphDataset(data_dir=val_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=min(cfg.batch_size, 32),
                              drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    # WandB
    if not cfg.no_wandb:
        wandb.init(config=cfg, project="sem_vp_train_graph", reinit=False)

    # MODEL AND OPTIMIZER
    pred_model = RecurrentGCN(node_features=8, out_features=2).to(cfg.device)
    optimizer = None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=cfg.lr)
        optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                         min_lr=1e-6, verbose=True)

    # LOSSES
    mse_loss = nn.MSELoss()  # TODO  how to measure "distance" between 6D poess?

    # MAIN LOOP
    for epoch in range(0, cfg.epochs):

        # train
        print(f'\nTraining (epoch: {epoch+1} of {cfg.epochs})')
        train_iter(cfg, train_loader, pred_model, optimizer, mse_loss)

        # eval
        print("Validating...")
        eval_loss = eval_iter(cfg, valid_loader, pred_model, mse_loss)
        optimizer_scheduler.step(eval_loss)
        eval_loss = eval_loss.item()
        print(f"Validation loss (mean over entire validation set): {eval_loss}")

        # save model if last epoch improved indicator loss
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
            torch.save(pred_model, best_model_path)
            print(f"Minimum indicator loss (mse) reduced -> model saved!")

        # visualize current model performance every nth epoch, using eval mode and validation data.
        if epoch % 10 == 9 and not cfg.no_vis:
            print("Saving visualizations...")
            out_filenames = vis_iter(cfg, val_data, pred_model)

            if not cfg.no_wandb:
                log_vids = {f"vis_{i}": wandb.Video(out_fn, fps=3,format="gif") for i, out_fn in enumerate(out_filenames)}
                wandb.log(log_vids, commit=False)

        # final bookkeeping
        if not cfg.no_wandb:
            wandb.log({"mse": eval_loss}, commit=True)

    # TESTING
    print("\nTraining done, testing best model...")
    cfg.models = [best_model_path]
    cfg.full_evaluation = True
    test_data = SynpickGraphDataset(data_dir=test_dir, num_frames=cfg.vid_total_length, step=cfg.vid_step,
                                     allow_overlap=cfg.vid_allow_overlap)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    test_loss = eval_iter(cfg, test_loader, pred_model, mse_loss).item()
    print(f"Test loss: {test_loss}")
    if not cfg.no_wandb:
        wandb.log({"test_mse": test_loss}, commit=True)
        wandb.finish()

    print("Testing done, bye bye!")
    return test_loss

# ==============================================================================

def train_iter(cfg, loader, pred_model, optimizer, mse_loss):

    loop = tqdm(loader)
    for _, data in enumerate(loop):

        loss = 0
        for _, snapshot in enumerate(data):

            snapshot = snapshot.to(cfg.device)
            y_hat = pred_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += mse_loss(y_hat, snapshot.y)

        # bwd
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pred_model.parameters(), 100)
        optimizer.step()

        # bookkeeping
        loop.set_postfix(loss=loss.item())
        #loop.set_postfix(mem=torch.cuda.memory_allocated())


def eval_iter(cfg, loader, pred_model, mse_loss):

    pred_model.eval()
    loop = tqdm(loader)
    eval_losses = []

    with torch.no_grad():

        for _, data in enumerate(loop):

            loss = 0
            for _, snapshot in enumerate(data):

                snapshot = snapshot.to(cfg.device)
                y_hat = pred_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                loss += mse_loss(y_hat, snapshot.y).detach()

            eval_losses.append(loss)
            loop.set_postfix(loss=loss.item())
            #loop.set_postfix(mem=torch.cuda.memory_allocated())

    eval_loss = torch.stack(eval_losses).mean()
    pred_model.train()

    return eval_loss


def vis_iter(cfg, dataset, pred_model, num_vis=10, test=False):

    from moviepy.editor import ImageSequenceClip

    out_fn_template = "vis_{}_test.gif" if test else "vis_{}.gif"
    out_fn_g_template = "vis_{}_t{}_test.png" if test else "vis_{}_t{}.png"
    out_filenames = []

    pred_model.eval()
    vis_idx = np.random.choice(len(dataset), num_vis, replace=False)

    with torch.no_grad():
        for g, idx in enumerate(vis_idx):
            data = dataset[idx]
            out_g_filenames = []
            for t, snapshot in enumerate(data):

                snapshot = snapshot.to(cfg.device)
                y_hat = pred_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                out_g_fn = os.path.join(cfg.out_dir, out_fn_g_template.format(g, t))
                out_g_filenames.append(out_g_fn)
                draw_synpick_pred_and_gt(snapshot, y_hat, out_g_fn)

            clip = ImageSequenceClip(out_g_filenames, fps=3)
            out_fn = os.path.join(cfg.out_dir, out_fn_template.format(g))
            out_filenames.append(out_fn)
            clip.write_gif(out_fn, fps=3)
            for out_g_fn in out_g_filenames:
                os.remove(out_g_fn)

    pred_model.train()

    return out_filenames