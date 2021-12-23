import argparse
from pathlib import Path
import torch

from vp_suite.training import train
from vp_suite.models.factory import AVAILABLE_MODELS
from vp_suite.dataset.factory import AVAILABLE_DATASETS
from vp_suite.measure.loss_provider import AVAILABLE_LOSSES
from vp_suite.utils.utils import timestamp, StoreDictKeyPair

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="vp-suite")
    parser.add_argument("--program", type=str, default="train_pred", choices=["train_pred", "test_pred"],
                        help="train_pred: train a prediction model, test_pred: test one or more prediction models.")
    parser.add_argument("--no-train", action="store_true", help="If specified, the training loop is skipped")
    parser.add_argument("--no-val", action="store_true", help="If specified, the validation loop is skipped")
    parser.add_argument("--no-vis", action="store_true", help="If specified, no visualizations are generated")
    parser.add_argument("--vis-every", type=int, default=10, help="Visualize predictions after every Nth epoch")
    parser.add_argument("--no-wandb", action="store_true", help="If specified, does not invoke WandB for logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--model-type", type=str, choices=AVAILABLE_MODELS, default="st_phy",
                        help="Which prediction model arch to use (See TODO for a full list of available models)")  # TODO full list of available models
    parser.add_argument("--tensor-value-range", type=float, nargs=2, default=[0.0, 1.0],
                        help="Two values specifying the value range of the pytorch tensors processed by the model")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--losses-and-scales", action=StoreDictKeyPair, default={"mse": 1.0}, nargs="+", metavar="KEY=VAL",
                        help="All mentioned measures will be calculated/logged. Also, if the scale value is non-zero, "
                             "they will be scaled and added to the loss for backprop.")
    parser.add_argument("--context-frames", type=int, default=10, help="Number of input frames for predictor")
    parser.add_argument("--pred-frames", type=int, default=10, help="Number of frames predicted from input")
    parser.add_argument("--val-rec-criterion", type=str, choices=AVAILABLE_LOSSES, default="mse",
                        help="Reconstruction loss type to use for performance assessment during validation")

    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--out-dir", type=str, default=f"out/{timestamp('train')}",
                        help="Output path for results (models, visualizations...)")
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-seq-step", type=int, default=1,
                        help="Use every nth frame of the video sequence. If n=1, no frames are skipped.")
    parser.add_argument("--use-actions", action="store_true",
                        help="If specified, do action-conditional learning if both the dataset and the model allow it")

    parser.add_argument("--use-optuna", action="store_true",
                        help="If specified, starts an optuna hyperparameter optimization.")
    parser.add_argument("--optuna-n-trials", type=int, default=30,
                        help="Number of hyperopt trials for optuna")

    # parse args and adjust as needed
    cfg = parser.parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.total_frames = cfg.context_frames + cfg.pred_frames
    cfg.opt_direction = "maximize" if AVAILABLE_LOSSES[cfg.val_rec_criterion].bigger_is_better else "minimize"

    # enter the training loop
    if cfg.use_optuna:
        from functools import partial
        try:
            import optuna
        except ImportError:
            raise ImportError("Importing optuna failed -> install it or use the code without the 'use-optuna' flag.")
        optuna_program = partial(train, cfg=cfg)
        study = optuna.create_study(directions=cfg.opt_direction)
        study.optimize(optuna_program, n_trials=cfg.optuna_n_trials)
    else:
        train(cfg=cfg)