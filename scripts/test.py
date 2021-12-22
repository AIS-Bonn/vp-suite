import argparse
from pathlib import Path
import torch

from vp_suite.testing import test
from vp_suite.models.factory import AVAILABLE_MODELS
from vp_suite.dataset.factory import AVAILABLE_DATASETS
from vp_suite.evaluation.loss_provider import AVAILABLE_LOSSES
from vp_suite.utils.utils import timestamp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="vp-suite")

    parser.add_argument("--model-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--no-vis", action="store_true", help="If specified, no visualizations are generated")
    parser.add_argument("--no-wandb", action="store_true", help="If specified, does not invoke WandB for logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (python, numpy, pytorch)")
    parser.add_argument("--tensor-value-range", type=float, nargs=2, default=[0.0, 1.0],
                        help="Two values specifying the value range of the pytorch tensors processed by the model")  # TODO infer from model cfg!
    parser.add_argument("--out-dir", type=str, default=f"out/{timestamp('test')}",
                        help="Output path for results (models, visualizations...)")
    parser.add_argument("--tensor-value-range", type=float, nargs=2, default=[0.0, 1.0],
                        help="Two values specifying the value range of the pytorch tensors processed by the model")

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--data-seq-step", type=int, default=1,
                        help="Use every nth frame of the video sequence. If n=1, no frames are skipped.")
    parser.add_argument("--context-frames", type=int, default=10, help="Number of input frames for predictor")  # TODO optionally infer from model cfg!
    parser.add_argument("--pred-frames", type=int, default=10, help="Number of frames predicted from input") # TODO optionally infer from model cfg!
    parser.add_argument("--test-metrics", type=str, nargs="+", default="all")
    parser.add_argument("--mini-test", action="store_true",
                        help="If specified, the models are tested only on a few datapoints of the test set")

    parser.add_argument("--use-actions", action="store_true",
                        help="If specified, do action-conditional learning if both the dataset and the model allow it")  # TODO infer from model cfg!

    # parse args and adjust as needed
    cfg = parser.parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.total_frames = cfg.context_frames + cfg.pred_frames

    # test the models
    test(cfg)