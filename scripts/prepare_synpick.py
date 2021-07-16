import sys, time, shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import random

def prepare_synpick(path, seed=42):

    # TODO shuffle episodes only?

    random.seed(seed)

    train_path = path / "train"
    test_path = path / "test"

    # get all training image FPs for rgb and semseg
    rgbs = sorted(train_path.glob("*/rgb/*.jpg"))
    segs = sorted(train_path.glob("*/class_index_masks/*.png"))
    r_s = list(zip(rgbs, segs))
    random.shuffle(r_s)
    rgbs, segs = list(zip(*r_s))

    # random split 'training' images into train and val (7:1)
    lr, ls = len(rgbs), len(segs)
    assert lr == ls
    cut = int(lr * 7 / 8)
    train_rgbs, val_rgbs = rgbs[:cut], rgbs[cut:]
    train_segs, val_segs = segs[:cut], segs[cut:]

    # get all testing testing image FPs
    test_rgbs = sorted(test_path.glob("*/rgb/*.jpg"))
    test_segs = sorted(test_path.glob("*/class_index_masks/*.png"))
    test_r_s = list(zip(test_rgbs, test_segs))
    random.shuffle(test_r_s)
    test_rgbs, test_segs = list(zip(*test_r_s))

    all_fps = [train_rgbs, train_segs, val_rgbs, val_segs, test_rgbs, test_segs]

    # prepare file copying
    out_path = Path("data").absolute() / "synpick_{}".format(int(time.time()))
    out_path.mkdir(parents=True)
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
            resized_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            ep_number = ''.join(filter(str.isdigit, fp.parent.parent.stem)).zfill(6)
            out_fp = "{}_{}{}".format(ep_number, fp.stem, ".".join(fp.suffixes))
            cv2.imwrite(str((op / out_fp).absolute()), resized_img)

if __name__ == '__main__':
    synpick_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
        prepare_synpick(synpick_path, seed)
    else:
        prepare_synpick(synpick_path)