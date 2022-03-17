[![PyPi](https://img.shields.io/pypi/v/vp-suite?color=blue&style=for-the-badge)](https://pypi.org/project/vp-suite/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/vp-suite?style=for-the-badge&color=blue)](https://pepy.tech/project/vp-suite)
[![License Badge](https://img.shields.io/github/license/AIS-Bonn/vp-suite?color=brightgreen&style=for-the-badge)](https://github.com/AIS-Bonn/vp-suite#license)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/AIS-Bonn/vp-suite/docs_pages_workflow?label=Docs&style=for-the-badge)](https://ais-bonn.github.io/vp-suite/)
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/vp-suite) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/vp-suite)

### Introduction

_Video prediction ('VP') is the task of predicting future frames given some context frames._

Like with most Computer Vision sub-domains, scientific contributions in this field exhibit a high variance in the following aspects:
- **Training protocol** (dataset usage, when to backprop, value ranges etc.)
- **Technical details of model implementation** (deep learning framework, package dependencies etc.) 
- **Benchmark selection and execution** (this includes the choice of dataset, number of context/predicted frames, skipping frames in the observed sequences etc.)
- **Evaluation protocol** (metrics chosen, variations in implementation/reduction modes, different ways of creating visualizations etc.)

Furthermore, while many contributors nowadays do share their code, seemingly minor missing parts such as dataloaders etc. make it much harder to assess, compare and improve existing models.  

This repo aims at providing a suite that facilitates scientific work in the subfield, providing standardized yet customizable solutions for the aspects mentioned above. This way, validating existing VP models and creating new ones hopefully becomes much less tedious.


### Installation

**Requires `pip` and `python >= 3.6` (code is tested with version `3.8`).**

<details>
<summary><b>From PyPi</b></summary>
   
   ```
   pip install vp-suite
   ```
</details>
<details>
<summary><b>From source</b></summary>
   
   ```
   pip install git+https://github.com/Flunzmas/vp-suite.git
   ```
</details>
<details>
<summary><b>If you want to contribute</b></summary>
   
   ```
   git clone https://github.com/Flunzmas/vp-suite.git
   cd vp-suite
   pip install -e .[dev]
   ```
</details>
<details>
<summary><b>If you want to build docs</b></summary>
   
   ```
   git clone https://github.com/Flunzmas/vp-suite.git
   cd vp-suite
   pip install -e .[doc]
   ```
</details>

### Usage

<details>
<summary><b>Changing save location</b></summary>

When using this package for the first time, the save location for datasets, 
models and logs is set to `<installation_dir>/vp-suite-data`. 
If you'd like to change that, simply run:
  
```
python vp_suite/resource/set_run_path.py
```

This script changes your save location and migrates any existing data.
</details>

<details>
<summary><b>Training models</b></summary>
   
```python
from vp_suite import VPSuite

# 1. Set up the VP Suite.
suite = VPSuite()

# 2. Load one of the provided datasets.
#    They will be downloaded automatically if no downloaded data is found.
suite.load_dataset("MM")  # load moving MNIST dataset from default location

# 3. Create a video prediction model.
suite.create_model('convlstm-shi')  # create a ConvLSTM-Based Prediction Model.
   
# 4. Run the training loop, optionally providing custom configuration.
suite.train(lr=2e-4, epochs=100)
```

This code snippet will train the model, log training progress to your [Weights & Biases](https://wandb.ai) account,
save model checkpoints on improvement and generate and save prediction visualizations.
</details>

<details>
<summary><b>Evaluating models</b></summary>

```python
from vp_suite import VPSuite

# 1. Set up the VP Suite.
suite = VPSuite()

# 2. Load one of the provided datasets in test mode.
#    They will be downloaded automatically if no downloaded data is found.
suite.load_dataset("MM", split="test")  # load moving MNIST dataset from default location

# 3. Get the filepaths to the models you'd like to test and load the models
model_dirs = ["out/model_foo/", "out/model_bar/"]
for model_dir in model_dirs:
    suite.load_model(model_dir, ckpt_name="best_model.pth")
    
# 4. Test the loaded models on the loaded test sets.
suite.test(context_frames=5, pred_frames=10)
```

This code will evaluate the loaded models on the loaded dataset (its test portion, if avaliable), 
creating detailed summaries of prediction performance across a customizable set of metrics.
The results as well as prediction visualizations are saved and logged to [Weights & Biases](https://wandb.ai).

_Note 1: If the specified evaluation protocol or the loaded dataset is incompatible with one of the models, 
this will raise an error with an explanation._

_Note 2: By default, a [CopyLastFrame](https://github.com/AIS-Bonn/vp-suite/blob/main/vp_suite/models/model_copy_last_frame.py) 
baseline is also loaded and tested with the other models._
</details>

<details>
<summary><b>Hyperparameter Optimization</b></summary>

This package uses [optuna](https://github.com/optuna/optuna) to provide hyperparameter optimization functionalities.
The following snippet provides a full example:

```python
import json
from vp_suite import VPSuite
from vp_suite.defaults import SETTINGS

suite = VPSuite()
suite.load_dataset(dataset="KTH")  # select dataset of choice
suite.create_model(model_id="lstm")  # select model of choice
with open(str((SETTINGS.PKG_RESOURCES / "optuna_example_config.json").resolve()), 'r') as cfg_file:
    optuna_cfg = json.load(cfg_file)
# optuna_cfg specifies the parameters' search intervals and scales; modify as you wish.
suite.hyperopt(optuna_cfg, n_trials=30, epochs=10)
```
This code e.g. will run 30 training loops (called _trials_ by optuna), producing a trained model for each hyperparameter configuration and writing the hyperparameter configuration of the best performing run to the console.

_Note 1: For hyperopt, visualization, logging and model checkpointing is minimized to reduce IO strain._

_Note 2: Despite optuna's trial pruning capabilities, running a high number of trials might still take a lot of time.
In that case, consider e.g. reducing the number of training epochs._

 Use `no_wandb=True`/`no_vis=True`
 if you want to log outputs to the console instead/not generate and save visualizations.

</details>

**Notes:**

- Use `VPSuite.list_available_models()` and `VPSuite.list_available_datasets()` to get an overview of which models and datasets are currently covered by the framework.
- All training, testing and hyperparametrization calls can be heavily configured (adjusting training hyperparameters, logging behavior etc, ...).
  For a comprehensive list of all adjustable run configuration parameters see the documentation of the `vp_suite.defaults` package.

### Customization

This package is designed with quick extensibility in mind. See the sections below for how to add new components 
(models, model blocks, datasets or measures).

<details>
<summary><b>New Models</b></summary>

1. Create a file `<your name>.py` in the folder `vp_suite/models`.
2. Create a class that derives from `vp_suite.base.base_model.VideoPredictionModel` and override/specify new constants you need.
3. Write your model code or import existing code so that the superclass interface is still served.
   If desired, you can implement a custom training/evaluation loop iteration `train_iter()`/`eval_iter()` 
   that gets called instead of the default training/evaluation loop iteration.
4. Register your model in the `MODEL_CLASSES` dictionary of `vp_suite/models/__init__.py`, giving it a key that can be used by the suite.
   By now, you should be able to create an instance of your model with `VPSuite.create_model()` and train it on a dataset with `VPSuite.train()`.

</details>

<details>
<summary><b>New Model Blocks</b></summary>

1. Create a file `<your name>.py` in the folder `vp_suite/model_blocks`.
2. Create a class that derives from `vp_suite.base.base_model_block.ModelBlock` and override/specify new constants you need.
3. Write your model block code or import existing code so that the superclass interface is still served.
4. If desired, add a local import of your model block to `vp_suite/model_blocks/__init__.py` (this registers the model block package-wide).

</details>

<details>
<summary><b>New Datasets</b></summary>

1. Create a file `<your name>.py` in the folder `vp_suite/datasets`.
2. Create a class that derives from `vp_suite.base.base_dataset.BaseVPDataset` and override/specify new constants you need.
3. Write your dataset code or import existing code so that the superclass interface is served. 
   If it's a public dataset, consider implementing methods to automatically download it.
4. Register your dataset in the `DATASET_CLASSES` dict of `vp_suite/dataset/__init__.py`, giving it a key that can be used by the suite.
   By now, you should be able to load your dataset with `VPSuite.load_dataset()` and train models on it with `VPSuite.train()`.

</details>

<details>
<summary><b>New measures (losses and/or metrics)</b></summary>

1. Create a new file `<your name>.py` in the folder `vp_suite/measure`, containing your loss or metric.
2. Make `vp_suite.base.base_measure.BaseMeasure` its superclass and override/implement all needed implementations and constants.
3. Register the measure in the `METRIC_CLASSES` dict of `vp_suite/measure/__init__.py` and, if it can also be used as a loss, in the `LOSS_CLASSES` dict.

</details>

**Notes:**

- If you omit the docstring for a particular attribute/method/field, the docstring of the base class is used for documentation.
- If implementing components that originate from publications/public repositories, please override the corresponding constants to specify the source!
  Additionally, if you want to write automated tests checking implementation equality, 
  have a look at how `tests/test_impl_match.py` fetches the tests of `tests/test_impl_match/` and executes these tests.
- Basic unit tests for models, datasets and measures are executed on all registered models - 
  you don't need to write such basic tests for your custom components! 
  Same applies for documentation: The tables that list available components are filled automatically.


### Contributing

This project is always open to extension! It grows especially powerful with more models and datasets, so if you've made your code work on custom models/datasets/metrics/etc., feel free to submit a merge request!

Other kinds of contributions are also very welcome - just check the open issues on the
[tracker](https://github.com/AIS-Bonn/vp-suite/issues) or open up a new issue there.

#### Unit Testing

When submitting a merge request, please make sure all tests run through (execute from root folder):
```
python -m pytest --runslow --cov=vp_suite
```
_Note: this is the easiest way to run all tests [without import hassles](https://docs.pytest.org/en/latest/explanation/pythonpath.html#invoking-pytest-versus-python-m-pytest).
You will need to have `vp-suite` installed in development move, though ([see here](#installation))._

#### API Documentation

The official API documentation is updated automatically upon push to the main branch.
If you want to build the documentation locally, make sure you've installed the package [accordingly](#installation)
and execute the following:
```
cd docs/
bash assemble_docs.sh
```

### Acknowledgements

- Project structure is inspired by [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).
- Sphinx-autodoc templates are inspired by the [QNET](https://github.com/mabuchilab/QNET) repository.

All other sources are acknowledged in the documentation of the respective point of usage (to the best of our knowledge).

### License

This project comes with an [MIT License](https://github.com/AIS-Bonn/vp-suite/blob/main/LICENSE), except for the following components:

- Module `vp_suite.measure.fvd.pytorch_i3d` (Apache 2.0 License, taken and modified from [here](https://github.com/piergiaj/pytorch-i3d))

### Disclaimer

I do not host or distribute any dataset. For all provided dataset functionality, I trust you have the permission to download and use the respective data. 