import argparse, time
from pathlib import Path
import torch
import optuna

from scripts.train_pred import train as train_pred_model
from scripts.train_seg import train as train_seg_model
from scripts.test_pred import test_pred_models
from scripts.visualize_4_way import visualize_4_way
from models.prediction.pred_model_factory import test_all_models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SYNPICK_CLASSES = 22
PROGRAMS = {
    "train_seg": train_seg_model,
    "train_pred": train_pred_model,
    "test_pred": test_pred_models,
    "4way_vis": visualize_4_way,
    "test_factory": test_all_models
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SEMANTIC VIDEO PREDICTION")
    parser.add_argument("--program", type=str, default="train_pred",
                        choices=["train_seg", "train_pred", "test_pred", "4way_vis", "test_factory"],
                        help="Specifies the program to run: training a semantic segmentation model (train_seg), "
                             "training a prediction model (train_pred), testing one or more prediction models (test_pred), "
                             "doing a 4-way visualization comparison on trained seg. and pred. models (4way_vis), "
                             "testing inference on all available model architectures (test_factory)")
    parser.add_argument("--no-train", action="store_true", help="If specified, the training loop is skipped")
    parser.add_argument("--no-vis", action="store_true", help="If specified, the visualization loops are skipped")
    parser.add_argument("--no-wandb", action="store_true", help="If specified, skips usage of WandB for logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--include-gripper", action="store_true", help="If specified, gripper is included in masks")
    parser.add_argument("--include-actions", action="store_true",
                        help="If specified, use gripper deltas for action-conditional learning")
    parser.add_argument("--pred-arch", type=str, choices=["unet", "lstm", "st_lstm", "copy", "phy", "st_phy"],
                        default="st_lstm", help="Which prediction model arch to use")
    parser.add_argument("--pred-mode", type=str, choices=["rgb", "colorized", "mask"], default="rgb",
                        help="Which kind of data to train/test on")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-existing-model", action="store_true")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=DEVICE)
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output path for results (models, visualizations...)")
    parser.add_argument("--dataset-classes", type=int, default=SYNPICK_CLASSES,
                        help="Number of object classes in dataset (applies to semantic segmentation/prediction only)")
    parser.add_argument("--vid-step", type=int, default=2, help="For video dataset, use only every nth frame")
    parser.add_argument("--vid-allow-overlap", action="store_true",
                        help="If specified, allows frames to appear in multiple valid video sequence datapoints")
    parser.add_argument("--vid-input-length", type=int, default=10, help="Number of input frames for predictor")
    parser.add_argument("--vid-pred-length", type=int, default=10, help="Number of frames predicted from input")
    parser.add_argument("--models", type=str, nargs="*", default=[], help="Pred. test only: path to pred. models")
    parser.add_argument("--full-test", action="store_true", help="If specified, tests models on the whole test set")

    parser.add_argument("--mse-loss-scale", type=float, default=1.0)
    parser.add_argument("--l1-loss-scale", type=float, default=1.0)
    parser.add_argument("--smoothl1-loss-scale", type=float, default=0.0)
    parser.add_argument("--lpips-loss-scale", type=float, default=0.0)
    parser.add_argument("--fvd-loss-scale", type=float, default=0.0)
    parser.add_argument("--calc-zero-loss-scales", action="store_true", help="if specified, also calculates loss scores"
                                                                             "for those losses not used for backprop")
    parser.add_argument("--pred-val-criterion", type=str, choices=["mse", "fvd", "bce"], default="fvd",
                        help="Loss to use for determining if validated model has become 'better' and should be saved")

    # model-specific hyperparameters
    # seg.UNet
    parser.add_argument("--seg-unet-features", nargs="+", type=int, default=[64, 128, 256, 512])
    # pred.UNet
    parser.add_argument("--pred-unet-features", nargs="+", type=int, default=[8, 16, 32, 64])
    # pred.ST_LSTM
    parser.add_argument("--pred-st-enc-channels", type=int, default=64)
    parser.add_argument("--pred-st-num-layers", type=int, default=3)
    parser.add_argument("--pred-st-inflated-action-dim", type=int, default=3)
    parser.add_argument("--pred-st-decouple-loss-scale", type=float, default=100.0)
    parser.add_argument("--pred-st-rec-loss-scale", type=float, default=0.1)
    # pred.Phy
    parser.add_argument("--pred-phy-kernel-size", nargs=2, type=int, default=(7, 7))
    parser.add_argument("--pred-phy-enc-channels", type=int, default=49)
    parser.add_argument("--pred-phy-moment-loss-scale", type=float, default=1.0)

    parser.add_argument("--use-optuna", action="store_true")


    # parse args and adjust as needed
    cfg = parser.parse_args()
    if cfg.out_dir is None:
        timestamp = int(1000000 * time.time())
        cfg.out_dir = f"out/{timestamp}_{cfg.program}"
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.vid_total_length = cfg.vid_input_length + cfg.vid_pred_length

    # run the specified program
    program = PROGRAMS[cfg.program]
    if cfg.use_optuna:
        from functools import partial
        optuna_program = partial(program, cfg=cfg)
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(optuna_program, n_trials=30)
    else:
        program(cfg=cfg)