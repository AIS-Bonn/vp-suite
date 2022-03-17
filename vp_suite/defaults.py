r"""
This module contains the Settings class, which contains constants;
and pre-set program configurations, such as default run config parameters.
"""
import json
from pathlib import Path

from vp_suite.utils.utils import get_public_attrs, timed_input


class _PackageSettings:
    r"""
    This class contains package constants.
    """

    PKG_ROOT_PATH = Path(__file__).parent.parent  # The root path of the vp-suite repository.
    PKG_SRC_PATH = Path(__file__).parent  #: The path of the vp_suite package (not the installable vp-suite package, but the subfolder that actually contains the code).
    PKG_RESOURCES = PKG_SRC_PATH / "resources"  #: The resources path of the vp_suite package.
    LOCAL_CONFIG_FP: str = str((PKG_RESOURCES / "local_config.json").resolve())  #: Default file location for the local install config.
    DEFAULT_RUN_PATH = PKG_ROOT_PATH / "vp-suite-data"  #: The fallback run path.
    RUN_PATH = None  #: This path is the base path for all artifacts that are saved during vp-suite usage (i.e. models, datasets and logs).

    # obtain the RUN_PATH from the package installation config file
    try:
        with open(LOCAL_CONFIG_FP, "r") as _install_config_file:
            RUN_PATH = Path(json.load(_install_config_file)["run_path"])
    except FileNotFoundError as e:
        RUN_PATH = DEFAULT_RUN_PATH
        with open(LOCAL_CONFIG_FP, "w") as _install_config_file:
            json.dump({"run_path": str(RUN_PATH.resolve())}, _install_config_file)

    OUT_PATH = RUN_PATH / "output"  #: The path where trained models and corresponding visualizations will be saved.
    DATA_PATH = RUN_PATH / "data"  #: The path where downloaded data and datasets will be stored.
    WANDB_PATH = RUN_PATH  #: The path for logging run information with Weights and Biases.


class DefaultRunConfig:
    r"""
    This class holds the default run configuration parameters, specifying behaviour
    during training and testing. All parameters can be overridden by supplementing
    keyword args in the corresponding training/testing call.
    """
    no_train: bool = False  #: If set to True, the training loop is skipped.
    no_val: bool = False  #: If set to True, the validation loop is skipped during the training procedure. Instead the best model is saved after every epoch.
    no_vis: bool = False  #: If set to True, no visualizations are generated during training/testing.
    no_wandb: bool = False  #: If set to True, don't log the run data to Weights and Biases.
    vis_every: int = 10  #: After this many training epochs, model predictions on randomly sampled validation sequences are visualized and saved (if `no_vis` is not set to False).
    n_vis: int = 5  #: Number of visualizations generated each time the model is used for visualization.
    vis_mode: str = "gif"  #: Specifies how to save the generated visualization videos.
    vis_compare: bool = False  #: If set to True, during testing, also generate visualization figures where the predicted frames of all tested models are laid out side-by-side.
    vis_context_frame_idx = None  #: If not None, during testing, this parameter specifies which context frame to include in the visualization figure that lays out the predictions of all models.
    seed: int = 42  #: The seed for all random number generators (python, numpy, pytorch) used throughout training/testing.
    lr: float = 0.0001  #: The learning rate for the models.
    epochs: int = 1000000  #: Number of epochs the model is trained before finalizing the training procedure. By default, this is set to a large number to let the training run terminate by time-outing.
    max_training_hours: float = 48  #: Maximum number of training hours before finalizing the training procedure. When the training time is exceeded, the current training iteration is continued but becomes the last training iteration.
    batch_size: int = 32  #: The batch size used for training.
    losses_and_scales: dict = {"mse": 1.0}  #: A dictionary where the keys denote all losses that should be calculated and logged during training, and their corresponding values denote the factor with which to multiply and add these losses to the overall loss used for backpropagation.
    val_rec_criterion: str = "mse"  #: The measure that is used to determine the model quality during validation. Every time the resulting measurement is improved, the current model snapshot is saved as the current 'best model'.
    metrics = ["mse", "lpips", "psnr", "ssim"]  #: A list of the metrics used for testing. If instead of a list, "all" is specified, all mavailable metrics are calculated.
    context_frames: int = 10  #: The number of context frames given to the prediction models. Also used in determining the needed sequence length for dataset usage.
    pred_frames: int = 10  #: The number of frames the prediction model shall predict. Also used in determining the needed sequence length for dataset usage.
    seq_step: int = 1  #: Sequences taken from the dataset use every Nth frame, where N is this value (Default value is 1, meaning that every frame is taken for the sequence).
    use_actions: bool = False  #: If set to True, and the model supports actions, and the dataset contains actions, these actions will be used by the model for prediction.
    out_dir = None  #: A file path for the output directory of the model. If none is specified, creates a suitable directory at runtime.


SETTINGS = _PackageSettings()  #: A settings instance that can be imported by other modules. It contains program-internal settings such as save paths (for the default values please see the source code).
DEFAULT_RUN_CONFIG = get_public_attrs(DefaultRunConfig())  #: A dictionary containing the default run configuration specified in the :class:`DefaultRunConfig` class.
