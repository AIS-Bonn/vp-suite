r"""
These tests run on the simplest models and datasets to speed up things, as here it is about testing the VP suite itself.
The testing of all models and datasets is done in test_models.py and test_datasets.py.
"""

import tempfile
import json
import pytest
from vp_suite import VPSuite
from vp_suite.defaults import SETTINGS

import torchvision.transforms as TF


dataset1 = "KTH"
dataset2 = "P101"
model1 = "lstm"
model2 = "phy"


@pytest.mark.slow
def test_creating_saving_loading_model():
    suite = VPSuite()
    suite.load_dataset(dataset_id=dataset1)
    suite.create_model(model_id=model1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        suite.train(epochs=0, no_wandb=True, out_dir=tmpdirname)
        suite.clear_models()
        suite.load_model(tmpdirname, ckpt_name="final_model.pth")


@pytest.mark.slow
def test_full_training_procedure():
    suite = VPSuite()
    suite.load_dataset(dataset_id=dataset1)
    suite.create_model(model_id=model1)
    suite.train(batch_size=16, epochs=1, vis_every=1, no_wandb=True)


@pytest.mark.slow
def test_hyperopt():
    suite = VPSuite()
    suite.load_dataset(dataset_id=dataset1)
    suite.create_model(model_id=model1)
    with open(str((SETTINGS.PKG_RESOURCES / "optuna_example_config.json").resolve()), 'r') as cfg_file:
        optuna_cfg = json.load(cfg_file)
    suite.hyperopt(optuna_cfg, batch_size=16, n_trials=1, epochs=1, no_wandb=True)


@pytest.mark.slow
def test_full_testing_single_dataset_single_model():
    suite = VPSuite()
    suite.load_dataset(dataset_id=dataset1, split="test")
    suite.create_model(model_id=model1)
    suite.test(context_frames=4, pred_frames=6, no_wandb=True)


@pytest.mark.slow
def test_brief_testing_multi_dataset_multi_model():
    suite = VPSuite()
    suite.load_dataset(dataset_id=dataset1, split="test")
    crop = TF.RandomCrop(size=1024)
    suite.load_dataset(dataset_id=dataset2, split="test", crop=crop, img_size=(64, 64))
    suite.create_model(model_id=model1)
    suite.create_model(model_id=model2, temporal_dim=3)
    suite.test(brief_test=True, context_frames=4, pred_frames=6, no_wandb=True)
