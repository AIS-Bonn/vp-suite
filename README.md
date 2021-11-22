# Semantic Video Prediction

This repo contains code to train models that can be used for semantic video prediction:
- Semantic Segmentation models (e.g. a UNet-like Model)
- Video Prediction models (on RGB data or Semantic Segmentation information) (using mostly recurrent or convolutional models)
- Object interaction/feature prediction models (using graph neural networks)

Furthermore, it also provides the code to prepare existing [SynPick](http://ais.uni-bonn.de/datasets/synpick/)
datasets for the training routined provided.

## Installation and Usage

### Installation

We'll be using pip and the Anaconda environment manager.
The code has been tested with Python 3.8, PyTorch 1.9 and CUDA 10.2.

```
git clone git@git.ais.uni-bonn.de:boltres/semantic-video-prediction.git
cd semantic-video-prediction
conda env create -f environment.yml
```

This yml file excludes `PyTorch`, `pytorch-geometric`, and `pytorch-geometric-temporal` because getting matching versions is easier done by hand:
1. Get a version of PyTorch that suits your needs (e.g. CUDA version) from [here](https://pytorch.org/get-started/locally/).
2. Follow [this](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html) doc to install matching versions of `pytorch-geometric` and `pytorch-geometric-temporal`.

### Usage

All scripts are run using `argparse`,
so feel free to check the descriptions for all available parameters using the `-h` option. 

**Prepare the SynPick dataset for training** (The resulting datasets for semantic
segmentation (`img`), video prediction (`vid`) and object interaction learning (`graph`)
will be put into the `data` folder and its directory name will include the timestamp of creation):

```
python scripts/prepare_synpick --in-path <path_to_synpick_dataset> --all
```

**Train in one of the available training modes** (`train_seg, train_pred, train_graph`):

```
python run.py --program <your_training_mode> --data-dir <path_to_prepared_dataset>
```


