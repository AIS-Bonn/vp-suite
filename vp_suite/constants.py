from pathlib import Path

PKG_ROOT_PATH = Path(__file__).parent
PKG_RESOURCES = PKG_ROOT_PATH / "resources"
RUN_PATH = Path("vp-suite")
OUT_PATH = RUN_PATH / "output"
DATA_PATH = RUN_PATH / "data"
WANDB_PATH = RUN_PATH