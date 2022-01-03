# vp-suite: All things Video Prediction.

_Video prediction ('VP') is the task of predicting future frames given some context frames._

Like with most Computer Vision sub-domains, scientific contributions in this field exhibit a high variance in the following aspects:
- **Training protocol** (dataset usage, when to backprop, value ranges etc.)
- **Technical details of model implementation** (deep learning framework, package dependencies etc.) 
- **Benchmark selection and execution** (this includes the choice of dataset, number of context/predicted frames, skipping frames in the observed sequences etc.)
- **Evaluation protocol** (metrics chosen, variations in implementation/reduction modes, different ways of creating visualizations etc.)

Furthermore, while many contributors nowadays do share their code, seemingly minor missing parts such as dataloaders etc. make it much harder to assess, compare and improve existing models.  

This repo aims at providing a suite that facilitates scientific work in the subfield, providing standardized yet customizable solutions for the aspects mentioned above. This way, validating existing VP models and creating new ones hopefully becomes much less tedious.

### Installation

The code has been tested with Python 3.8, CUDA 11.3 and PyTorch 1.10. We'll use the conda package manager 

```
git clone git@github.com:Flunzmas/vp-suite.git
cd vp-suite
conda env create -f environment.yml
conda activate vp-suite
```

### Usage

#### Basic Usage

All scripts are run using `argparse`,
so feel free to check the descriptions for all available parameters of the provided entry points using the `-h` option. 

- Train a VP model:  `python scripts/train.py --dataset <dataset_ID> --data-dir <path/to/dataset/folder>`
- Test one or more (pre-)trained VP models on some dataset: `python scripts/test.py --model-dirs <path/to/model1/folder> <...> --dataset <dataset_ID> --data-dir <path/to/dataset/folder>`

#### Creating new VP models/Integrating existing external models 

1. Create a file `model_<your name>.py` in the folder `vp_suite/models`.
2. Create a class that derives from `vp_suite.models.base_model.VideoPredictionModel` and override the things you need.
3. Write your model code or import existing code so that the superclass interface is still served. If desired, you can implement a custom training loop iteration `train_iter(cfg, train_loader, optimizer, loss_provider, epoch)` that gets called instead of the default training loop iteration.
4. Write tests for your model (`test/test_models.py`) and register it in the `pred_models` dict of `vp_suite/models/factory.py`.
5. Check training performance on different datasets, fix things and contribute to the project 😊

#### Training on new datasets

1. Create a file `dataset_<your name>.py` in the folder `vp_suite/dataset`.
2. Create a class that derives from `vp_suite.dataset.base_dataset.BaseVPDataset` and override the things you need.
3. Write your dataset code or import existing code so that the superclass interface is served and the dataset initialization with `vp_suite/dataset/factory.py` still works.
4. Write tests for your dataset (`test/test_dataset.py`) and register it in the `dataset_classes` dict of `vp_suite/dataset/factory.py`.
5. Check training performance with different models, fix things and contribute to the project 😊

#### Custom losses, metrics and optimization

1. Create a new file in `vp_suite/measure`, containing your loss or metric.
2. Make `vp_suite.measure.base_measure.BaseMeasure` its superclass and provide all needed implementations and attributes.
3. Register the measure in the `METRICS` dict of `vp_suite/measure/metric_provider.py` and, if it can also be used as a loss, in the `LOSSES` dict of `vp_suite/measure/loss_provider.py`.
4. Write tests for the measure (`test/test_measures.py`).
5. Check training/evaluation performance with different models and datasets, fix things and contribute to the project 😊

### Contributing

This project is always open to extension! If you're adding models, datasets or measures, just be sure to subclass the according base classes and write tests so that the code can be used by others.

Other kinds of contributions are also welcome.

### Citing

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

This project comes with an [MIT License](https://github.com/Flunzmas/vp-suite/blob/main/LICENSE).
