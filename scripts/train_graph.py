import os, time, random
from pathlib import Path

import wandb

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from dataset.graph.synpick_graph import SynpickGraphDataset
from models.graph_pred.rgcn import RecurrentGCN

def train(trial=None, cfg=None):

    # PREPARATION pt. 1
    best_eval_loss = float("inf")
    best_model_path = str((Path(cfg.out_dir) / 'best_model.pth').resolve())
    num_classes = cfg.dataset_classes + 1 if cfg.include_gripper else cfg.dataset_classes

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
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.batch_size,
                              drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    # WandB
    if not cfg.no_wandb:
        wandb.init(config=cfg, project="sem_vp_train_graph", reinit=False)

    # MODEL AND OPTIMIZER
    pred_model = RecurrentGCN(node_features=8, out_features=7)  # TODO
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
        eval_loss = indicator_loss.item()
        print(f"Validation loss (mean over entire validation set): {eval_loss}")

        # save model if last epoch improved indicator loss
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
            torch.save(pred_model, best_model_path)
            print(f"Minimum indicator loss (mse) reduced -> model saved!")

        # visualize current model performance every nth epoch, using eval mode and validation data.
        if epoch % 10 == 9 and not cfg.no_vis:
            print("Saving visualizations...")
            print("TODO")
            pass

            #if not cfg.no_wandb:
            #    log_vids = {f"vis_{i}": wandb.Video(out_fn, fps=4,format="gif") for i, out_fn in enumerate(out_filenames)}
            #    wandb.log(log_vids, commit=False)

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
    test_loss = eval_iter(test_loader, pred_model, mse_loss).item()
    print(f"Test loss: {test_loss}")
    #if not cfg.no_wandb:
    #    wandb.log(test_metrics, commit=True)
    #    wandb.finish()

    print("Testing done, bye bye!")
    return test_loss

# ==============================================================================

def train_iter(cfg, loader, pred_model, optimizer, mse_loss):

    loop = tqdm(loader)
    for _, graph_temporal_signal in enumerate(loop):

        loss = 0
        for i, cur_graph in enumerate(graph_temporal_signal):
            cur_graph = cur_graph.to(cfg.device)
            predicted_feat = pred_model(cur_graph)
            loss += mse_loss(predicted_feat, cur_graph["y"])

        # bwd
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pred_model.parameters(), 100)
        optimizer.step()

        # bookkeeping
        loop.set_postfix(loss=loss.item())
        # loop.set_postfix(mem=torch.cuda.memory_allocated())


def eval_iter(cfg, loader, pred_model, mse_loss):

    pred_model.eval()
    loop = tqdm(loader)
    eval_losses = []

    with torch.no_grad():

        for _, frame_graphs in enumerate(loop):

            frame_graphs = [graph.to(cfg.device) for graph in frame_graphs]
            loss = 0
            for i in range(len(frame_graphs) - 1):
                cur_graph, next_graph = frame_graphs[i], frame_graphs[i + 1]
                predicted_feat = pred_model(cur_graph)
                loss += mse_loss(predicted_feat["features"], next_graph["features"][:-1]).detach()

            eval_losses.append(loss)
            loop.set_postfix(loss=loss.item())

    eval_loss = torch.stack(eval_losses).mean()
    pred_model.train()

    return eval_loss