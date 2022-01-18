[📚 **Link to Documentation** 📚](https://flunzmas-vp-suite.readthedocs.io/en/latest/)  [![Documentation Status](https://readthedocs.org/projects/flunzmas-vp-suite/badge/?version=latest)](https://flunzmas-vp-suite.readthedocs.io/en/latest/?badge=latest)


### Introduction

_Video prediction ('VP') is the task of predicting future frames given some context frames._

Like with most Computer Vision sub-domains, scientific contributions in this field exhibit a high variance in the following aspects:
- **Training protocol** (dataset usage, when to backprop, value ranges etc.)
- **Technical details of model implementation** (deep learning framework, package dependencies etc.) 
- **Benchmark selection and execution** (this includes the choice of dataset, number of context/predicted frames, skipping frames in the observed sequences etc.)
- **Evaluation protocol** (metrics chosen, variations in implementation/reduction modes, different ways of creating visualizations etc.)

Furthermore, while many contributors nowadays do share their code, seemingly minor missing parts such as dataloaders etc. make it much harder to assess, compare and improve existing models.  

This repo aims at providing a suite that facilitates scientific work in the subfield, providing standardized yet customizable solutions for the aspects mentioned above. This way, validating existing VP models and creating new ones hopefully becomes much less tedious.

### Installation (required: Python >= 3.8)

From PyPi: 
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vp-suite
```

From source:
```
pip install git+https://github.com/Flunzmas/vp-suite.git
```

### Usage

When using this package, a folder `vp-suite` is created in your current working directory/current path 
that will contain all downloaded data as well as run logs, outputs and trained models.

#### Training models

1. Set up the trainer:
2. Load one of the provided datasets (will be downloaded automatically) [create your own](#training-on-custom-datasets):
3. Create video prediction model (either from scratch or from a pretrained checkpoint, can be one of the provided models or your own):
4. Run the training loop, optionally providing custom configuration

```python
from vp_suite import VPSuite

suite = VPSuite()
suite.load_dataset("MM")  # load moving MNIST dataset from default location

model_checkpoint = ""  # Set to valid model path to load a checkpoint
if model_checkpoint != "":
   suite.load_model(model_checkpoint)
else:
   suite.create_model('lstm')  # create a ConvLSTM-Based Prediction Model
   
suite.train(lr=2e-4, epochs=100)
```

This will train the model, log training progress to the console (and optionally to [Weights & Biases](https://wandb.ai)),
save model checkpoints on improvement and, optionally, generate and save prediction visualizations.

#### Evaluating models

1. Set up the tester
2. Load one of the provided datasets or (will be downloaded automatically) [create your own](#training-on-custom-datasets)
3. Load the models you'd like to test (by default, a [CopyLastFrame](https://github.com/Flunzmas/vp-suite/blob/main/vp_suite/models/model_copy_last_frame.py) baseline is already loaded)
4. Run the testing on all models, optionally providing custom configuration of the evaluation protocol:

```python
from vp_suite import VPSuite

suite = VPSuite()
suite.load_dataset("MM")  # load moving MNIST dataset from default location

# get the filepaths to the models you'd like to test
model_dirs = ["out/model_foo/", "out/model_bar/"]
for model_dir in model_dirs:
    suite.load_model(model_dir, ckpt_name="best_model.pth")
suite.test(context_frames=5, pred_frames=10)
```

This code will evaluate the loaded models on the loaded dataset (its test portion, if avaliable), creating detailed summaries of prediction performance across a customizable set of metrics.
Optionally, the results as well as prediction visualizations can be saved and logged to [Weights & Biases](https://wandb.ai).

_Note: if the specified evaluation protocol or the loaded dataset is incompatible with one of the models, this will raise an error with an explanation._ 

#### Hyperparameter Optimization

This package uses [optuna](https://github.com/optuna/optuna) to provide hyperparameter optimization functionalities.
The following snippet provides a full example:

```python
import json
from vp_suite import VPSuite
from vp_suite.constants import PKG_RESOURCES

suite = VPSuite()
suite.load_dataset(dataset="KTH")  # select dataset of choice
suite.create_model(model_type="lstm")  # select model of choice
with open(str((PKG_RESOURCES / "optuna_example_config.json").resolve()), 'r') as cfg_file:
    optuna_cfg = json.load(cfg_file)
# optuna_cfg specifies the parameters' search intervals and scales; modify as you wish.
suite.hyperopt(optuna_cfg, trials=30, epochs=10)
```
This code e.g. will run 30 training loops (called _trials_ by optuna), producing a trained model for each hyperparameter configuration and writing the hyperparameter configuration of the best performing run to the console.

_Note 1: For hyperopt, visualization, logging and model checkpointing is minimized to reduce IO strain._

_Note 2: Despite optuna's trial pruning capabilities, running a high number of trials might still take a lot of time.
In that case, consider e.g. reducing the number of training epochs._

### Customization

While this package comes with a few pre-defined models/datasets/metrics etc. for your convenience, it was designed with quick extensibility in mind. See the sections below for how to add new models, datasets or metrics.

#### Creating new VP models or integrating existing external models 

1. Create a file `model_<your name>.py` in the folder `vp_suite/models`.
2. Create a class that derives from `vp_suite.models.base_model.VideoPredictionModel` and override the things you need.
3. Write your model code or import existing code so that the superclass interface is still served.
   If desired, you can implement a custom training loop iteration `train_iter(self, config, loader, optimizer, loss_provider, epoch)` that gets called instead of the default training loop iteration.
4. Check training performance on different datasets, fix things and contribute to the project 😊

#### Training on custom datasets

1. Create a file `dataset_<your name>.py` in the folder `vp_suite/dataset`.
2. Create a class that derives from `vp_suite.dataset.base_dataset.BaseVPDataset` and override the things you need.
3. Write your dataset code or import existing code so that the superclass interface is served and the dataset initialization with `vp_suite/dataset/factory.py` still works.
4. Register it in the `DATASET_CLASSES` dict of `vp_suite/dataset/__init__.py`.
5. Run pytest, check training performance with different models, fix things and contribute to the project 😊

#### Custom losses, metrics and optimization

1. Create a new file in `vp_suite/measure`, containing your loss or metric.
2. Make `vp_suite.measure.base_measure.BaseMeasure` its superclass and provide all needed implementations and attributes.
3. Register the measure in the `METRIC_CLASSES` dict of `vp_suite/measure/__init__.py` and, if it can also be used as a loss, in the `LOSS_CLASSES` dict.
4. Run pytest, check training/evaluation performance with different models and datasets, fix things and contribute to the project 😊

### Contributing

This project is always open to extension! It grows especially powerful with more models and datasets, so if you've made your code work on custom models/datasets/metrics/etc., feel free to submit a merge request!

Other kinds of contributions are also very welcome - just check the open issues on the
[tracker](https://github.com/Flunzmas/vp-suite/issues) or open up a new issue there.

When submitting a merge request, please make sure all tests run through (execute from root folder):
```
python -m pytest --runslow
```
_Note: this is the easiest way to run all tests [without import hassles](https://docs.pytest.org/en/latest/explanation/pythonpath.html#invoking-pytest-versus-python-m-pytest).
Omit the `runslow` argument to speed up testing by removing the tests for the complete training/testing procedure._

### Acknowledgements

Project structure is inspired by [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).

All other sources are acknowledged in the documentation of the respective point of usage (to the best of our knowledge).

### Citing

If you use this package/repository for your academic work, please consider citing it as follows:

```
@misc{vp_suite,
  Author = {Boltres, Andreas},
  Title = {vp-suite: A Framework for Training and Evaluating Video Prediction Models},
  Year = {2022},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/Flunzmas/vp-suite}}
}
```

### License stuffs

This project comes with an [MIT License](https://github.com/Flunzmas/vp-suite/blob/main/LICENSE), except for the following components:

- Module `vp_suite.measure.fvd.pytorch_i3d` (Apache 2.0 License, taken and modified from [here](https://github.com/piergiaj/pytorch-i3d))
