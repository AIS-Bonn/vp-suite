# Semantc Video Prediction

This repo contains code to train models for semantic segmentation and video prediction 
which can be combined to solve the task of semantic video prediction. Furthermore, it also 
provides the code to prepare existing [SynPick](http://ais.uni-bonn.de/datasets/synpick/)
datasets for the training routined provided.

All python scripts use the `argparse` argument parser, so you can run `<script_name>.py -h` to have a look at the
possible input arguments.

## Installation and Usage

### Installation

We'll be using the Anaconda environment manager.

```
git clone git@git.ais.uni-bonn.de:boltres/semantic-video-prediction.git
cd semantic-video-prediction
conda env create -f environment.yml
```

### Usage

**Prepare the SynPick dataset for training** (The resulting datasets for semantic
segmentation (`img`) and video prediction (`vid`) will be put into the `data`
folder and its directory name will include the timestamp of creation):

```
python scripts/prepare_synpick --in-path <path_to_synpick_dataset> --seed <your_seed>
```

**Train a semantic segmentation model** (UNet):

```
python train_seg_model.py --in-path data/synpick_img_<timestamp>
```


**Train a video prediction model** (e.g. a simple, UNet-inspired model):

```
python train_pred_model.py --in-path data/synpick_vid_<timestamp> --model unet
```

**To be continued...**


