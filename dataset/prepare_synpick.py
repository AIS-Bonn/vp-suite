import sys, time, shutil
from pathlib import Path
import argparse

from tqdm import tqdm
import numpy as np
import cv2
import random

TRAIN_VAL_SPLIT = (7/8)  # train:val  7:1

def prepare_synpick_graph(cfg):

    path = Path(cfg.in_path)
    seed = cfg.seed
    random.seed(seed)

    train_path = path / "train"
    test_path = path / "test"

    # get all training/validation FPs, split randomly by episode
    scene_gts = list(train_path.glob("*/scene_gt.json"))
    random.shuffle(scene_gts)
    cut = int(len(scene_gts) * TRAIN_VAL_SPLIT)
    train_scene_gts, val_scene_gts = sorted(scene_gts[:cut]), sorted(scene_gts[cut:])

    # get all testing FPs
    test_scene_gts = sorted(test_path.glob("*/scene_gt.json"))

    all_scene_gts = [train_scene_gts, val_scene_gts, test_scene_gts]
    out_path = Path("data").absolute() / f"graph_{path.stem}_{cfg.timestamp}"
    out_path.mkdir(parents=True)
    copy_synpick_scene_gts(all_scene_gts, out_path)

def prepare_synpick_img(cfg):

    path = Path(cfg.in_path)
    seed = cfg.seed

    random.seed(seed)

    train_path = path / "train"
    test_path = path / "test"

    # get all training image FPs for rgb and semseg
    rgbs = sorted(train_path.glob("*/rgb/*.jpg"))
    segs = sorted(train_path.glob("*/class_index_masks/*.png"))
    r_s = list(zip(rgbs, segs))
    random.shuffle(r_s)
    rgbs, segs = list(zip(*r_s))

    # random split 'training' images into train and val
    lr, ls = len(rgbs), len(segs)
    assert lr == ls
    cut = int(lr * TRAIN_VAL_SPLIT)
    train_rgbs, val_rgbs = rgbs[:cut], rgbs[cut:]
    train_segs, val_segs = segs[:cut], segs[cut:]

    # get all testing image FPs
    test_rgbs = sorted(test_path.glob("*/rgb/*.jpg"))
    test_segs = sorted(test_path.glob("*/class_index_masks/*.png"))
    test_r_s = list(zip(test_rgbs, test_segs))
    random.shuffle(test_r_s)
    test_rgbs, test_segs = list(zip(*test_r_s))

    all_fps = [train_rgbs, train_segs, val_rgbs, val_segs, test_rgbs, test_segs]
    out_path = Path("data").absolute() / f"img_{path.stem}_{cfg.timestamp}"
    out_path.mkdir(parents=True)

    copy_synpick_imgs(all_fps, out_path, cfg.resize_ratio)


def prepare_synpick_vid(cfg):

    path = Path(cfg.in_path)
    seed = cfg.seed

    random.seed(seed)

    train_path = path / "train"
    test_path = path / "test"

    # get all training image FPs for rgb
    rgbs = sorted(train_path.glob("*/rgb/*.jpg"))
    segs = sorted(train_path.glob("*/class_index_masks/*.png"))
    scene_gts = sorted(train_path.glob("*/scene_gt.json"))

    num_ep = int(Path(rgbs[-1]).parent.parent.stem) + 1
    train_eps = [i for i in range(num_ep)]
    random.shuffle(train_eps)
    cut = int(num_ep * TRAIN_VAL_SPLIT)
    train_eps = train_eps[:cut]

    # split rgb files into train and val by episode number only , as we need contiguous motions for video
    train_rgbs, val_rgbs, train_segs, val_segs, train_scene_gts, val_scene_gts = [], [], [], [], [], []
    for rgb, seg in zip(rgbs, segs):
        ep = int(Path(rgb).parent.parent.stem) + 1
        if ep in train_eps:
            train_rgbs.append(rgb)
            train_segs.append(seg)
        else:
            val_rgbs.append(rgb)
            val_segs.append(seg)

    for scene_gt in scene_gts:
        ep = int(Path(scene_gt).parent.stem) + 1
        if ep in train_eps:
            train_scene_gts.append(scene_gt)
        else:
            val_scene_gts.append(scene_gt)

    test_rgbs = sorted(test_path.glob("*/rgb/*.jpg"))
    test_segs = sorted(test_path.glob("*/class_index_masks/*.png"))
    test_scene_gts = sorted(test_path.glob("*/scene_gt.json"))

    all_img_fps = [train_rgbs, train_segs, val_rgbs, val_segs, test_rgbs, test_segs]
    all_scene_gts = [train_scene_gts, val_scene_gts, test_scene_gts]
    out_path = Path("data").absolute() / f"vid_{path.stem}_{cfg.timestamp}"
    out_path.mkdir(parents=True)

    copy_synpick_imgs(all_img_fps, out_path, cfg.resize_ratio)
    copy_synpick_scene_gts(all_scene_gts, out_path)


def copy_synpick_imgs(all_fps, out_path, resize_ratio):

    # prepare and execute file copying
    all_out_paths = [(out_path / "train" / "rgb"), (out_path / "train" / "masks"),
                     (out_path / "val" / "rgb"), (out_path / "val" / "masks"),
                     (out_path / "test" / "rgb"), (out_path / "test" / "masks")]

    for op in all_out_paths:
        op.mkdir(parents=True)

    # copy files to new folder structure
    for fps, op in zip(all_fps, all_out_paths):
        for fp in tqdm(fps, postfix=op.parent.stem + "/" + op.stem):
            fp = Path(fp)
            img = cv2.imread(str(fp.absolute()), cv2.IMREAD_COLOR if "jpg" in fp.suffix else cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
            ep_number = ''.join(filter(str.isdigit, fp.parent.parent.stem)).zfill(6)
            out_fp = "{}_{}{}".format(ep_number, fp.stem, ".".join(fp.suffixes))
            cv2.imwrite(str((op / out_fp).absolute()), resized_img)


def copy_synpick_scene_gts(all_fps, out_path):

    # prepare and execute file copying
    all_out_paths = [(out_path / "train" / "scene_gt"), (out_path / "val" / "scene_gt"),
                     (out_path / "test" / "scene_gt")]

    for op in all_out_paths:
        op.mkdir(parents=True)

    # copy files to new folder structure
    for fps, op in zip(all_fps, all_out_paths):
        for fp in fps:
            ep_number = ''.join(filter(str.isdigit, fp.parent.stem)).zfill(6)
            out_fp = op / "{}_{}{}".format(ep_number, fp.stem, ".".join(fp.suffixes))
            print(fp, out_fp)
            shutil.copyfile(fp, out_fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="prepare_synpick")
    parser.add_argument("--in-path", type=str, help="directory for synpick data")
    parser.add_argument("--seed", type=int, default=42, help="rng seed for train/val split")
    parser.add_argument("--all", action="store_true", help="prepare data for all datasets")
    parser.add_argument("--graph", action="store_true", help="prepare data for graph dataset")
    parser.add_argument("--img", action="store_true", help="prepare data for image dataset")
    parser.add_argument("--vid", action="store_true", help="prepare data for video dataset")
    parser.add_argument("--resize-ratio", type=float, default=0.25, help="Scale frame sizes by this amount")

    cfg = parser.parse_args()
    cfg.timestamp = int(time.time())

    if cfg.all:
        cfg.graph = cfg.img = cfg.vid = True
    if cfg.graph:
        print("Preparing graph dataset...")
        prepare_synpick_graph(cfg)
    if cfg.img:
        print("Preparing image dataset...")
        prepare_synpick_img(cfg)
    if cfg.vid:
        print("Preparing video dataset...")
        prepare_synpick_vid(cfg)