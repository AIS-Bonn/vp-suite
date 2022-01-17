import tempfile
import json
import pytest
from vp_suite import VPSuite
import vp_suite.constants as constants

"""
These tests run on the simplest models and datasets to speed up things, as here it is about testing the VP suite itself.
The testing of all models and datasets is done in test_models.py and test_datasets.py.
"""

dataset1 = "KTH"
dataset2 = "MM"
model1 = "non_conv"
model2 = "simplev1"


@pytest.mark.slow
def test_creating_saving_loading_model():
    suite = VPSuite()
    suite.load_dataset(dataset=dataset1)
    suite.create_model(model_type=model1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        suite.train(epochs=0, no_wandb=True, out_dir=tmpdirname)
        suite.clear_models()
        suite.load_model(tmpdirname, ckpt_name="final_model.pth")


@pytest.mark.slow
def test_full_training_procedure():
    suite = VPSuite()
    suite.load_dataset(dataset=dataset1)
    suite.create_model(model_type=model1)
    suite.train(batch_size=16, epochs=1, vis_every=1, no_wandb=True)


@pytest.mark.slow
def test_hyperopt():
    suite = VPSuite()
    suite.load_dataset(dataset=dataset1)
    suite.create_model(model_type=model1)
    with open(str((constants.PKG_RESOURCES / "optuna_example_config.json").resolve()), 'r') as cfg_file:
        optuna_cfg = json.load(cfg_file)
    suite.hyperopt(optuna_cfg, batch_size=16, n_trials=3, epochs=1, no_wandb=True)


@pytest.mark.slow
def test_full_testing_single_dataset_single_model():
    suite = VPSuite()
    suite.load_dataset(dataset=dataset1)
    suite.create_model(model_type=model1)
    suite.test(context_frames=4, pred_frames=6, no_wandb=True)


@pytest.mark.slow
def test_full_testing_multi_dataset_multi_model():
    suite = VPSuite()
    suite.load_dataset(dataset=dataset1)
    suite.load_dataset(dataset=dataset2)
    suite.create_model(model_type=model1)
    suite.create_model(model_type=model2)
    suite.test(brief_test=True, context_frames=4, pred_frames=6, no_wandb=True)
