import sys, time, shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np

def prepare_synpick(path, seed):

    train_path = path / "train"
    test_path = path / "test"

    # get all training image FPs for rgb, semseg, and those w/o bg
    np.random.seed(seed)
    rgbs = np.random.permutation(list(train_path.glob("*/rgb/*.jpg")))
    np.random.seed(seed)
    segs = np.random.permutation(list(train_path.glob("*/class_index_masks/*.png")))

    # random split 'training' images into train and val (7:1)
    lr, ls = len(rgbs), len(segs)
    assert lr == ls
    cut = int(lr * 7 / 8)
    train_rgbs, val_rgbs = rgbs[:cut], rgbs[cut:]
    train_segs, val_segs = segs[:cut], segs[cut:]

    # get all testing testing image FPs
    np.random.seed(seed)
    test_rgbs = np.random.permutation(list(test_path.glob("*/rgb/*.jpg")))
    np.random.seed(seed)
    test_segs = np.random.permutation(list(test_path.glob("*/class_index_masks/*.png")))

    all_fps = [test_rgbs, test_segs]

    # prepare file copying
    out_path = Path("out").absolute() / "data" / "synpick_{}".format(int(time.time()))
    out_path.mkdir(parents=True)
    all_out_paths = [(out_path / "train" / "rgb"), (out_path / "train" / "masks"),
                     (out_path / "val" / "rgb"), (out_path / "val" / "masks"),
                     (out_path / "test" / "rgb"), (out_path / "test" / "masks")]

    for op in all_out_paths:
        op.mkdir(parents=True)

    # copy files to new folder structure
    for fps, op in zip(all_fps, all_out_paths):
        for fp in tqdm(fps):
            fp = Path(fp)
            ep_number = ''.join(filter(str.isdigit, fp.parent.parent.stem)).zfill(6)
            out_fp = "{}_{}{}".format(ep_number, fp.stem, ".".join(fp.suffixes))
            shutil.copyfile(fp, op / out_fp)

if __name__ == '__main__':
    synpick_path = Path(sys.argv[1])
    seed = int(sys.argv[2])
    prepare_synpick(synpick_path, seed)