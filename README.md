# vp-suite: All things Video Prediction.

_Video prediction ('VP') is the task of predicting future frames given some context frames._

Like with most Computer Vision sub-domains, scientific contributions in this field exhibit a high variance in the following aspects:
- **Training protocol** (dataset usage, when to backprop, value ranges etc.)
- **Technical details of model implementation** (deep learning framework, package dependencies etc.) 
- **Benchmark selection and execution** (this includes the choice of dataset, number of context/predicted frames, skipping frames in the observed sequences etc.)
- **Evaluation protocol** (metrics chosen, variations in implementation/reduction modes, different ways of creating visualizations etc.)

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

- Train a VP model:  `TODO`
- Test one or more (pre-)trained VP models on some dataset: `TODO`

#### Creating new VP models/Integrating existing external models 

TODO

#### Training on new datasets

TODO

#### Custom training procedures

TODO

#### Custom losses, metrics and optimization

TODO

#### Custom logging

TODO

### Contributing

TODO

### Citing

TODO

### License stuffs

TODO
