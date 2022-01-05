import sys
sys.path.append(".")
import pytest
from vp_suite.training import Trainer
from vp_suite.models._factory import AVAILABLE_MODELS

@pytest.mark.slow
def test_all_models_on_kth():
    vp_trainer = Trainer()
    data_dir = "/home/data/datasets/video_prediction/kth_actions"
    vp_trainer.load_dataset(dataset="KTH", data_dir=data_dir)
    for model_type in AVAILABLE_MODELS:
        vp_trainer.create_model(model_type=model_type)
        vp_trainer.train(epochs=1, no_wandb=True)
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


if __name__ == '__main__':
    test_all_models_on_kth()