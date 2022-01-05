import os
import sys
sys.path.append(".")
import pytest
from vp_suite.trainer import Trainer
from vp_suite.tester import Tester
from vp_suite.models._factory import AVAILABLE_MODELS

@pytest.mark.slow
def test_all_models_on_kth():
    vp_trainer = Trainer()
    data_dir = "/home/data/datasets/video_prediction/kth_actions"
    vp_trainer.load_dataset(dataset="KTH", data_dir=data_dir)
    for model_type in AVAILABLE_MODELS:
        vp_trainer.create_model(model_type=model_type)
        vp_trainer.train(epochs=5, no_wandb=True)
    assert True  # test successful if execution reaches this line

@pytest.mark.slow
def test_all_datasets_on_non_conv():
    vp_trainer = Trainer()
    data_dirs = {
        "KTH": "/home/data/datasets/video_prediction/kth_actions",
        #"MM": "/home/data/datasets/video_prediction/moving_mnist/moving_mnist_frames20_digits2",
        "BAIR": "/home/data/datasets/video_prediction/bair_robot_pushing"
    }
    for dataset, data_dir in data_dirs.items():
        vp_trainer.load_dataset(dataset, data_dir=data_dir)
        vp_trainer.create_model(model_type="non_conv")
        vp_trainer.train(epochs=1, no_wandb=True, use_actions=True)
    assert True  # test successful if execution reaches this line

@pytest.mark.slow
def test_tester_on_BAIR():
    vp_tester = Tester()
    bair_data_dir = "/home/data/datasets/video_prediction/bair_robot_pushing"
    vp_tester.load_dataset("BAIR", data_dir=bair_data_dir)
    all_model_dirs = ["out/" + fp for fp in os.listdir("out") if fp.startswith("train")]
    vp_tester.load_models(all_model_dirs)
    vp_tester.test(context_frames=5, pred_frames=5, no_wandb=True, no_vis=True,
                   metrics=["mse", "ssim", "psnr", "lpips"])
    assert True  # test successful if execution reaches this line


if __name__ == '__main__':
    test_tester_on_BAIR()