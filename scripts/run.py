import argparse
from pathlib import Path
import torch

from vp_suite.train import train as train_pred_model
from scripts.test_models import test_pred_models
from vp_suite.models.model_factory import MODELS
from vp_suite.dataset.dataset_factory import SUPPORTED_DATASETS
from vp_suite.utils.utils import timestamp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROGRAMS = {
    "train_pred": train_pred_model,
    "test_pred": test_pred_models,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SEMANTIC VIDEO PREDICTION")
    parser.add_argument("--program", type=str, default="train_pred",
                        choices=["train_pred", "test_pred"],
                        help="train_pred: train a prediction model, test_pred: test one or more prediction models.")
    parser.add_argument("--no-train", action="store_true", help="If specified, the training loop is skipped")
    parser.add_argument("--no-val", action="store_true", help="If specified, the validation loop is skipped")
    parser.add_argument("--no-vis", action="store_true", help="If specified, no visualizations are generated")
    parser.add_argument("--vis-every", type=int, default=10, help="Visualize predictions after every Nth epoch")
    parser.add_argument("--no-wandb", action="store_true", help="If specified, does not invoke WandB for logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--model-type", type=str, choices=MODELS,
                        default="st_lstm", help="Which prediction model arch to use (See TODO for a full list of available models)")  # TODO full list of available models
    parser.add_argument("--tensor-value-range", type=float, nargs=2, default=[0.0, 1.0],
                        help="Two values specifying the value range of the pytorch tensors processed by the model")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=DEVICE)
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--out-dir", type=str, default=f"out/{timestamp()}",
                        help="Output path for results (models, visualizations...)")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS)

    parser.add_argument("--data-seq-step", type=int, default=1,
                        help="Use every nth frame of the video sequence. If n=1, no frames are skipped.")
    parser.add_argument("--context-frames", type=int, default=10, help="Number of input frames for predictor")
    parser.add_argument("--pred-frames", type=int, default=10, help="Number of frames predicted from input")

    parser.add_argument("--teacher-forcing-epochs", type=int, default=0,
                        help="Number of epochs in which a decaying teacher forcing ratio is used")
    parser.add_argument("--mse-loss-scale", type=float, default=1.0)
    parser.add_argument("--l1-loss-scale", type=float, default=0.0)
    parser.add_argument("--smoothl1-loss-scale", type=float, default=0.0)
    parser.add_argument("--lpips-loss-scale", type=float, default=0.0)
    parser.add_argument("--ssim-loss-scale", type=float, default=0.0)
    parser.add_argument("--psnr-loss-scale", type=float, default=0.0)
    parser.add_argument("--fvd-loss-scale", type=float, default=0.0)
    parser.add_argument("--calc-zero-loss-scales", action="store_true", help="if specified, also calculates loss scores"
                                                                             "for those losses not used for backprop")
    parser.add_argument("--train-rec-criterion", type=str, choices=LOSS_DESC, default=DEFAULT_LOSS_DESC,
                        help="Reconstruction loss type to use for optimization during training")  # TODO enable optional specification of loss scales
    parser.add_argument("--val-rec-criterion", type=str, choices=LOSS_DESC, default=DEFAULT_LOSS_DESC,
                        help="Reconstruction loss type to use for performance assessment during validation")
    parser.add_argument("--test-metrics")

    # dataset-specific arguments
    parser.add_argument("--include-gripper", action="store_true", help="If specified, gripper is included in masks")
    parser.add_argument("--include-actions", action="store_true",
                        help="If specified, use gripper deltas for action-conditional learning")
    parser.add_argument("--dataset-classes", type=int, default=SYNPICK_CLASSES,
                        help="Number of object classes in dataset (applies to semantic segmentation/prediction only)")

    # test arguemtns
    parser.add_argument("--models", type=str, nargs="*", default=[], help="Pred. test only: path to pred. models")
    parser.add_argument("--full-test", action="store_true", help="If specified, tests models on the whole test set")


    # model-specific hyperparameters
    # seg.UNet
    parser.add_argument("--seg-unet-features", nargs="+", type=int, default=[64, 128, 256, 512])
    # pred.UNet
    parser.add_argument("--pred-unet-features", nargs="+", type=int, default=[8, 16, 32, 64])
    # pred.ConvLSTM
    parser.add_argument("--pred-lstm-num-layers", type=int, default=3)
    parser.add_argument("--pred-lstm-kernel-size", nargs=2, type=int, default=(3, 3))
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

    parser.add_argument("--use-optuna", action="store_true",
                        help="If specified, starts an optuna hyperparameter optimization.")
    parser.add_argument("--optuna-n-trials", type=int, default=30,
                        help="Number of hyperopt trials for optuna")

    # parse args and adjust as needed
    cfg = parser.parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.total_frames = cfg.context_frames + cfg.pred_frames

    # run the specified program
    program = PROGRAMS[cfg.program]
    if cfg.use_optuna:
        from functools import partial
        try:
            import optuna
        except ImportError:
            raise ImportError("Importing optuna failed -> install it or use the code without the 'use-optuna' flag.")
        optuna_program = partial(program, cfg=cfg)
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(optuna_program, n_trials=cfg.optuna_n_trials)
    else:
        program(cfg=cfg)