import random, json, os, time
import warnings
from typing import List, Dict, Any
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from vp_suite.defaults import SETTINGS, DEFAULT_RUN_CONFIG
from vp_suite.utils.dataset_wrapper import VPDatasetWrapper
from vp_suite.datasets import DATASET_CLASSES
from vp_suite.base import VPModel
from vp_suite.models import MODEL_CLASSES, AVAILABLE_MODELS
from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.measure import LOSS_CLASSES
from vp_suite.measure.loss_provider import PredictionLossProvider
from vp_suite.measure.metric_provider import PredictionMetricProvider
from vp_suite.utils.visualization import visualize_vid, visualize_sequences
from vp_suite.utils.utils import timestamp
from vp_suite.utils.compatibility import check_model_and_data_compat, check_run_and_model_compat


class VPSuite:
    r"""
    This class is the main workbench of `vp-suite`, in the sense that upon instantiating this class you've got access to
    all the functionalities of `vp-suite`: Loading and using datasets, loading or instantiating new models and training
    or testing them on the loaded datasets.

    Attributes:
        device(str): A string specifying which device to work on ('cuda' for GPU usage or 'cpu' for CPU).
        datasets(List[:class:`VPDatasetWrapper`]): The loaded datasets.
        models(List[:class:`VPModel`]): The loaded/created models.
    """
    def __init__(self, device: str = "cuda"):
        r"""
        Instantiates the `VPSuite`, setting device and clearing loaded models and datasets.

        Args:
            device (str): A string identifying whether to use GPU or CPU.
        """
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.clear_models()
        self.clear_datasets()

    @property
    def training_sets(self):
        r"""
        Returns: A list of all loaded training sets (last one in list is last loaded training set).
        """
        return [d for d in self.datasets if d.is_training_set()]

    @property
    def test_sets(self):
        r"""
        Returns: A list of all loaded test sets (last one in list is last loaded test set).
        """
        return [d for d in self.datasets if d.is_test_set()]

    def clear_datasets(self):
        r"""
        Empties the list of loaded datasets.
        """
        self.datasets : List[VPDatasetWrapper] = []

    def clear_models(self):
        r"""
        Empties the list of loaded models.
        """
        self.models : List[VPModel] = []

    def load_dataset(self, dataset_id: str, split: str = "train", **dataset_kwargs):
        r"""
        Creates the dataset specified by given dataset id and appends it to `VPSuite`'s list of loaded datasets.
        
        Args:
            dataset_id (str): The string ID mapping of the desired dataset.
            split (str): This string specifies whether to load the dataset in training or testing mode.
            **dataset_kwargs (Any): Optional additional dataset configuration options.
        """
        # create dataset wrapper
        dataset_class = DATASET_CLASSES[dataset_id]
        dataset = VPDatasetWrapper(dataset_class, split, **dataset_kwargs)
        print(f"loaded dataset '{dataset.NAME}' from {dataset.data_dir} "
              f"(action size: {dataset.action_size})")

        # if seq information is specified, directly execute set_seq_len
        if any([k in dataset_kwargs.keys() for k in ["context_frames", "pred_frames", "seq_step"]]):
            context_frames = dataset_kwargs.pop("context_frames", DEFAULT_RUN_CONFIG["context_frames"])
            pred_frames = dataset_kwargs.pop("pred_frames", DEFAULT_RUN_CONFIG["pred_frames"])
            seq_step = dataset_kwargs.pop("seq_step", DEFAULT_RUN_CONFIG["seq_step"])
            dataset.set_seq_len(context_frames, pred_frames, seq_step)

        self.datasets.append(dataset)

    def download_dataset(self, dataset_id: str):
        r"""
        Downloads the dataset specified by given dataset ID.

        Args:
            dataset_id (str): The string ID mapping of the desired dataset.
        """
        dataset_class = DATASET_CLASSES[dataset_id]
        dataset_class.download_and_prepare_dataset()

    def list_available_datasets(self):
        r"""
        Prints a list of all available datasets and their corresponding string IDs.
        """
        for dataset_id, dataset_class in DATASET_CLASSES.items():
            print(f"'{dataset_id}': {dataset_class.NAME}")

    def list_available_models(self):
        r"""
        Prints a list of all available models and their corresponding string IDs.
        """
        for model_id, model_class in MODEL_CLASSES.items():
            print(f"'{model_id}': {model_class.NAME}")

    def load_model(self, model_dir: str, ckpt_name: str = "best_model.pth"):
        r"""
        Loads the model saved in the specified checkpoint file or the specified directory
         and appends it to `VPSuite`'s list of loaded models.

        Args:
            model_dir (str): Relative path to the directory containing the saved model.
            ckpt_name (str): File name of the saved model.
        """
        model_ckpt = os.path.join(model_dir, ckpt_name)
        model = torch.load(model_ckpt)
        model.model_dir = model_dir
        self._model_setup(model, loaded=True)

    def create_model(self, model_id: str, action_conditional: bool = False, **model_kwargs):
        r"""
        Creates the model specified by given string ID and appends it to `VPSuite`'s list of loaded models.

        Args:
            model_id (str): The string ID corresponding to your desired model.
            action_conditional (bool): If the model supports actions, this variable determines whether the model will actually use provided actions for prediction.
            **model_kwargs (Any): Optional additional model configuration options.
        """

        # parameter processing
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"invalid model type specified! Available model types: {AVAILABLE_MODELS}")

        model_class = MODEL_CLASSES[model_id]
        for param in model_class.REQUIRED_ARGS:
            if param not in model_kwargs.keys():
                print(f"model parameter '{param}' not specified -> trying to take from last loaded dataset...")
                if len(self.datasets) < 1:
                    raise ValueError(f"no dataset loaded to take parameter '{param}' from")
                param_val = self.datasets[-1].config.get(param, None)
                if param_val is None:
                    raise ValueError(f"dataset '{self.datasets[-1].NAME}' doesn't provide parameter '{param}', "
                                     f"so it has to be specified on model creation")
                model_kwargs.update({param: param_val})
        if action_conditional and not model_class.CAN_HANDLE_ACTIONS:
            warnings.warn("specified model can't handle actions -> argument 'action_conditional' set to False")
            action_conditional = False
        model_kwargs.update(action_conditional=action_conditional)

        # model creation
        model = model_class(self.device, **model_kwargs).to(self.device)
        self._model_setup(model)

    def _model_setup(self, model: VPModel, loaded: bool = False):
        r"""
        Internal model setup, also appending the model to the list of loaded models.

        Args:
            model (VPModel): The video prediction model.
            loaded (bool): Identifies whether the model has been loaded (=True) or newly created (=False).
        """
        ac_str = "(action-conditional)" if model.config["action_conditional"] else ""
        loaded_str = "loaded" if loaded else "created new"
        print(f"{loaded_str} model '{model.NAME}' {ac_str}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" - Model parameters (total / trainable): {total_params} / {trainable_params}")
        self.models.append(model)

    def _prepare_run(self, split: str = "train", **run_kwargs):
        r"""
        Prepares the upcoming run by fetching/setting configuration, models and datasets.

        Args:
            split (str): Determines whether this run is going to be a training/hyperopt run or a test run.
            **run_kwargs (Any): Optional specified run configuration parameters (will override the defaults).
        """
        if len(self.models) == 0:
            raise RuntimeError("No model available. Load a pretrained model "
                               "or create a new instance before starting training or test runs")
        if split == "train" and len(self.training_sets) == 0:
            raise ValueError("No training sets loaded. Load a dataset in training mode "
                             "before starting training or test runs")
        elif split == "test" and len(self.test_sets) == 0:
            raise ValueError("No test sets loaded. Load a dataset in test mode "
                             "before starting training or test runs")
        run_config = deepcopy(DEFAULT_RUN_CONFIG)

        # update config
        if not all([run_arg in run_config.keys() for run_arg in run_kwargs.keys()]):
            raise ValueError(f"Only the following run arguments are supported: {run_config.keys()}")
        run_config.update(run_kwargs)

        self._set_seeds(run_config["seed"])

        # opt. direction
        run_config["opt_direction"] = "maximize" if LOSS_CLASSES[run_config["val_rec_criterion"]].BIGGER_IS_BETTER \
            else "minimize"

        return run_config

    def _set_seeds(self, seed: int):
        r"""
        Sets the general rng seeds for Python, numpy and PyTorch.

        Args:
            seed (int): The random seed for the rng's.

        Warning: This should remain the only code location where the general rng seeds are set!
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset_rng(self, seed: int):
        r"""
        Resets the RNG for all datasets and rng.

        Args:
            seed (int): The random seed for the rng's.
        """
        self._set_seeds(seed)
        for dataset in self.datasets:
            dataset.reset_rng()

# ===== TRAINING ================================================================

    def _prepare_training(self, dataset_idx: int, model_idx: int, **run_kwargs):
        r"""
        Prepares a single dataset and a single model for training.

        Args:
            dataset_idx (int): The list index of the dataset that should be used for training.
            model_idx (int) The list index of the model that should be trained on.
            **run_kwargs (Any): Optional specified run configuration parameters (will override the defaults).
        """
        run_config = self._prepare_run("train", **run_kwargs)

        try:
            dataset: VPDatasetWrapper = self.training_sets[dataset_idx]
            model: VPModel = self.models[model_idx]
        except IndexError:
            raise ValueError("given indices for model and/or dataset are invalid")

        # prepare dataset
        dataset.set_seq_len(run_config["context_frames"], run_config["pred_frames"], run_config["seq_step"])
        assert dataset.is_ready, "dataset is not ready even though set_seq_len has just been called"

        # compat checks: run <--> model; model <--> dataset
        check_run_and_model_compat(model, run_config)
        _, _ = check_model_and_data_compat(model, dataset, strict_mode=True)

        return model, dataset, run_config

    def train(self, trial=None, dataset_idx: int = -1, model_idx: int = -1, **run_kwargs):
        r"""
        Executes a training run on a single model and a single dataset.
        After preparation, the main training loops run training epochs until the max number of epochs
        or the time limit is reached.

        During each epoch: 1. the whole dataset is iterated through in (model-specific)
        training iterations that employ forward and backward passes. 2. Each model is validated on the whole validation
        set and saved if it improved its performance on that set. 3. Every few epochs, prediction visualizations are
        created and saved to the disk. 4. Current model performance is logged.

        Args:
            trial (Any): If calling this function within a hyperparameter optimization run, this object cantains the necessary parameters. Otherwise, it's None.
            dataset_idx (int): The list index of the dataset that should be used for training.
            model_idx (int) The list index of the model that should be trained on.
            **run_kwargs (Any): Optional specified run configuration parameters (will override the defaults).

        Returns: The best obtained validation loss (the corresponding model is saved as 'best_model.pt').
        """
        # PREPARATION
        model, dataset, run_config = self._prepare_training(dataset_idx, model_idx, **run_kwargs)
        train_data, val_data = dataset.train_data, dataset.val_data
        train_loader = DataLoader(train_data, batch_size=run_config["batch_size"], shuffle=True, num_workers=4,
                                  drop_last=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        best_val_loss = float("inf")

        # re-use model_dir of pre-loaded/pre-initialized models if no out_dir has been specified
        if run_config["out_dir"] is None and model.model_dir is not None:
            print(f"Using existing model save location ({model.model_dir})...")
            out_path = Path(model.model_dir)
        else:
            out_dir = run_config["out_dir"] or SETTINGS.OUT_PATH / timestamp('train')  # fetch default if None
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            model.model_dir = str(out_path.resolve())

        best_model_path = str((out_path / 'best_model.pth').resolve())
        with_training = model.TRAINABLE and not run_config["no_train"]
        with_validation = not run_config["no_val"]
        with_wandb = not run_config["no_wandb"]

        # HYPERPARAMETER OPTIMIZATION
        optuna_config = run_config.get("optuna", None)
        using_optuna = trial is not None and isinstance(optuna_config, dict)
        if using_optuna:
            for param, p_dict in optuna_config.items():
                if "choices" in p_dict.keys():
                    if param == "model_type":
                        warnings.warn(f"hyperopt across model and dataset parameters is not yet supported "
                                      f"-> using {model.NAME}")
                    run_config[param] = trial.suggest_categorical(param, p_dict["choices"])
                else:
                    suggest = trial.suggest_int if p_dict["type"] == "int" else trial.suggest_float
                    log_scale = p_dict.get("scale", "uniform") == "log"
                    if log_scale:
                        run_config[param] = suggest(param, p_dict["min"], p_dict["max"], log=log_scale)
                    else:
                        step = p_dict.get("step", 1)
                        run_config[param] = suggest(param, p_dict["min"], p_dict["max"], step=step)

        # assemble and save combined configuration
        config: Dict[str, Any] = {**run_config, **model.config, **dataset.config, "device": self.device,
                                  "model_name": model.NAME, "dataset_name": dataset.NAME}
        save_config = {"run": run_config, "model": model.config,
                       "dataset": dataset.config, "device": self.device}
        with open(str((out_path / 'run_cfg.json').resolve()), "w") as cfg_file:
            json.dump(save_config, cfg_file, indent=4,
                      default=lambda o: str(o) if callable(getattr(o, "__str__", None)) else '<not serializable>')

        # WandB
        if with_wandb:
            wandb_reinit = using_optuna and trial.number > 0
            wandb.init(config=config, project="vp-suite-training",
                       dir=str(SETTINGS.WANDB_PATH.resolve()), reinit=wandb_reinit)

        # OPTIMIZER
        optimizer, optimizer_scheduler = None, None
        if with_training:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=run_config["lr"])
            optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                             min_lr=1e-6, verbose=True)

        # LOSSES AND MEASUREMENT
        loss_provider = PredictionLossProvider(config)
        if config['val_rec_criterion'] not in config['losses_and_scales']:
            raise ValueError(f"Validation criterion '{config['val_rec_criterion']}' has to be "
                             f"one of the chosen losses: {list(config['losses_and_scales'].keys())}")
        if config["opt_direction"] == "maximize":
            def loss_improved(cur_loss, best_loss): return cur_loss > best_loss
        else:
            def loss_improved(cur_loss, best_loss): return cur_loss < best_loss

        # --- MAIN LOOP ---
        training_timeout = time.time() + config["max_training_hours"] * 3600
        for epoch in range(0, run_config["epochs"]):
            print(f"\nEpoch: {epoch+1} of {config['epochs']}")

            # train
            if with_training:
                print("Training...")
                model.train_iter(config, train_loader, optimizer, loss_provider, epoch)
            else:
                print("Skipping training loop.")

            # eval
            val_losses = dict()
            if with_validation:
                print("Validating...")
                val_losses, indicator_loss = model.eval_iter(config, val_loader, loss_provider)
                if with_training:
                    optimizer_scheduler.step(indicator_loss)
                print("Validation losses (mean over entire validation set):")
                for k, v in val_losses.items():
                    print(f" - {k}: {v}")

                # save model if last epoch improved indicator loss
                cur_val_loss = indicator_loss.item()
                if loss_improved(cur_val_loss, best_val_loss):
                    best_val_loss = cur_val_loss
                    torch.save(model, best_model_path)
                    print(f"Minimum indicator loss ({config['val_rec_criterion']}) reduced -> model saved!")
            else:
                print("Skipping validation loop and simply saving current model as the 'best' model.")
                torch.save(model, best_model_path)

            # visualize current model performance every nth epoch, using eval mode and validation data.
            if (epoch+1) % config["vis_every"] == 0 and not config["no_vis"]:
                print("Saving visualizations...")
                vis_out_dir = out_path / f"vis_ep_{epoch+1:03d}"
                vis_out_dir.mkdir(exist_ok=True)  # overrides existing visualizations (e.g. from previous runs)
                vis_idx = np.random.choice(len(val_data), config["n_vis"], replace=False)
                visualize_vid(val_data, config["context_frames"], config["pred_frames"], model,
                              config["device"], vis_out_dir, vis_idx, config["vis_mode"])

                if with_wandb:
                    vid_filenames = sorted(os.listdir(str(vis_out_dir)))
                    log_vids = {fn: wandb.Video(str(vis_out_dir / fn), fps=4, format=fn.split(".")[-1])
                                for i, fn in enumerate(vid_filenames)}
                    wandb.log(log_vids, commit=False)

            # final bookkeeping
            if with_validation and with_wandb:
                wandb.log(val_losses, commit=True)
            if time.time() > training_timeout:
                print("Maximum training time exceeded, leaving training loop...")
                break

        # finishing training by saving final model and returning best performance on validation set
        print("\nTraining done, cleaning up...")
        torch.save(model, str((out_path / 'final_model.pth').resolve()))
        wandb.finish()
        return best_val_loss  # return best validation loss for hyperparameter optimization

    def hyperopt(self, optuna_config: dict, n_trials: int = 30, dataset_idx: int = -1, model_idx: int = -1,
                 **run_kwargs):
        r"""
        Executes a hyperparameter optimization process using the bayesian optimization framework optuna.
        A pre-defined number of training runs ("trials") are executed sequentially on the same model and datasets
        under varying (hyperparameter) configurations. After the optimization, the best configuration is printed to the
        terminal.

        Args:
            optuna_config (dict): Optuna run configuration, specifies the search space (parameters and their search ranges) of the optimization.
            n_trials (int): The number of training runs executed within the optimization.
            dataset_idx (int): The list index of the dataset that should be used for training.
            model_idx (int) The list index of the model that should be trained on.
            **run_kwargs (Any): Optional specified run configuration parameters (will override the defaults). Apply to all runs within the optimizaion.
        """
        from functools import partial
        from vp_suite.utils.utils import check_optuna_config
        run_config = self._prepare_run(**run_kwargs)
        check_optuna_config(optuna_config)
        run_config["optuna"] = optuna_config
        try:
            import optuna
        except ImportError:
            raise ImportError("Importing optuna failed -> install it or use the code without the 'use-optuna' flag.")
        optuna_program = partial(self.train, dataset_idx=dataset_idx, model_idx=model_idx, **run_kwargs)
        study = optuna.create_study(direction=run_config["opt_direction"])
        study.optimize(optuna_program, n_trials=n_trials)
        print("\nHyperparameter optimization complete. Best performing parameters:")
        for k, v in study.best_params.items():
            print(f" - {k}: {v}")

# ===== TESTING ================================================================

    def _prepare_testing(self, **run_kwargs):
        r"""
        Prepares all loaded datasets and models for testing

        Args:
            **run_kwargs (Any): Optional specified run configuration parameters (will override the defaults).
        """
        run_config = self._prepare_run("test", **run_kwargs)

        # prepare datasets for testing
        test_sets: List[VPDatasetWrapper] = self.test_sets
        for test_set in test_sets:
            test_set.set_seq_len(run_config["context_frames"], run_config["pred_frames"], run_config["seq_step"])
            assert test_set.is_ready, "test set is not ready even though set_seq_len has just been called"

        # compat-check run arguments against models and skip incompatible models
        test_models = []
        for model in self.models:
            try:
                check_run_and_model_compat(model, run_config)
                test_models.append(model)
            except ValueError as e:
                print(f"skipping test of model '{model.NAME}' because of incompatibility with run config: {str(e)}")

        # compat-check loaded models against each dataset and skip incompatible model-dataset pairs
        model_lists_all_test_sets = []
        for test_set in test_sets:
            test_set_model_list = []
            for model in test_models:
                try:
                    # model-data compat check returns adapters if discrepancies can be bridged for testing
                    preprocessing, postprocessing = check_model_and_data_compat(model, test_set)
                    test_set_model_list.append((model, preprocessing, postprocessing, []))
                except ValueError as e:
                    print(f"skipping test of model '{model.NAME}' on dataset '{test_set.NAME}' "
                          f"because of incompatibility: {str(e)}")
            model_lists_all_test_sets.append(test_set_model_list)

            # add baseline copy model (doesn't need checks)
            clf_baseline = CopyLastFrame().to(self.device)
            test_set_model_list.append((clf_baseline, nn.Identity(), nn.Identity(), []))

        test_sets_and_model_lists = zip(test_sets, model_lists_all_test_sets)
        return test_sets_and_model_lists, run_config

    def _test_on_dataset(self, model_info_list, dataset: VPDatasetWrapper, run_config: dict, brief_test: bool):
        r"""
        Tests all models on a single dataset. Iterates through the whole dataset, retrieving the predictions of all
        specified models on each data point and calculating pre-specified performance metrics.

        Args:
            model_info_list (Any): A list containing the models to be tested as well as other needed information/objects such as adapters.
            dataset (VPDatasetWrapper): The dataset to be tested on.
            run_config (dict): The run configuration used for testing. For all unspecified run configuration parameters, the default configuration is used.
            brief_test (bool): If specified, only a brief sample check is done instead of fully iterating over the test set.
        """
        # PREPARATION
        test_data = dataset.test_data
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
        if len(test_loader) < 1:
            raise RuntimeError("loaded dataset does not contain any data (len < 1)")
        test_mode = "brief" if brief_test else "full"
        eval_length = min(len(test_loader), 10) if test_mode == "brief" else len(test_loader)

        # assemble and save combined configuration
        config: Dict[str, Any] = {**run_config, **dataset.config, "device": self.device, "dataset_name": dataset.NAME}
        with_wandb = not config["no_wandb"]

        # evaluation / metric calc.
        context_frames = config["context_frames"]
        pred_frames = config["pred_frames"]
        iter_loader = iter(test_loader)
        with torch.no_grad():
            metric_provider = PredictionMetricProvider(config)

            for _ in tqdm(range(eval_length)):
                data = next(iter_loader)

                for (model, preprocess, postprocess, model_metrics_per_dp) in model_info_list:
                    input, target, actions = model.unpack_data(data, config)
                    input = preprocess(input)  # test format to model format
                    model.eval()
                    if getattr(model, "use_actions", False):
                        pred, _ = model(input, pred_frames=pred_frames, actions=actions)
                    else:
                        pred, _ = model(input, pred_frames=pred_frames)
                    model.train()
                    pred = postprocess(pred)  # model format to test format
                    cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                    model_metrics_per_dp.append(cur_metrics)

        # save visualizations
        timestamp_test = timestamp('test')
        vis_out_dir = SETTINGS.OUT_PATH / timestamp_test
        vis_out_dir.mkdir()
        if not config["no_vis"]:
            print(f"Saving visualizations for trained models...")
            vis_idx = np.random.choice(len(test_data), config["n_vis"], replace=False)
            if test_data.ON_THE_FLY:  # if data is generated on-the-fly, reset dataset's RNG because the loaders could have mingled with it
                self.reset_rng(config["seed"])

            models = [m_info[0] for m_info in model_info_list]
            if config["vis_compare"]:
                vis_context_frame_idx = config["vis_context_frame_idx"] or list(range(context_frames))
            else:
                vis_context_frame_idx = None
            visualize_sequences(test_data, context_frames, pred_frames, models, config["device"],
                                vis_out_dir, vis_idx, vis_context_frame_idx, config["vis_mode"])

        # log or display metrics
        if eval_length > 0:
            wandb_full_suffix = f"{test_mode} test"
            for i, (model, _, _, model_metrics_per_dp) in enumerate(model_info_list):
                # model_metrics_per_dp: list of N lists of F metric dicts (called D).
                # each D_{n,f} contains all calculated metrics for datapoint 'n' and a prediction horizon of 'f' frames.
                # -> Aggregate these metrics over all n, keeping the specific metrics/prediction horizons separate
                datapoint_range = range(len(model_metrics_per_dp))
                frame_range = range(len(model_metrics_per_dp[0]))
                metric_keys = model_metrics_per_dp[0][0].keys()
                mean_metric_dicts = [
                    {metric_key: np.mean(
                        [model_metrics_per_dp[dp_i][frame][metric_key] for dp_i in datapoint_range]
                    ) for metric_key in metric_keys}
                    for frame in frame_range
                ]

                # Log model to WandB
                if with_wandb:
                    print("Logging test results to WandB for all models...")
                    wandb.init(config={"test_mode": test_mode, "model_dir": model.model_dir},
                               project="vp-suite-testing", name=f"{model.NAME} ({wandb_full_suffix})",
                               dir=str(SETTINGS.WANDB_PATH.resolve()), reinit=(i > 0))
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        wandb.log({"pred_frames": f+1, **mean_metric_dict})
                    if not config["no_vis"] and model.model_dir is not None:
                        vid_filenames = [fn for fn in sorted(os.listdir(str(vis_out_dir)))
                                         if fn.split(".")[-1] in ["mp4", "gif"]]
                        model_log_vids = {fn: wandb.Video(str(vis_out_dir / fn), fps=4, format=fn.split(".")[-1])
                                          for i, fn in enumerate(vid_filenames)}
                        wandb.log(model_log_vids)
                    if i == len(model_info_list) - 1:
                        wandb.finish()

                # Per-model/per-pred.-horizon console log of the metrics
                else:
                    print(f"\n{model.NAME} (path: {model.model_dir}): ")
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        print(f"pred_frames: {f + 1}")
                        for (k, v) in mean_metric_dict.items():
                            print(f" -> {k}: {v}")

    def test(self, brief_test=False, **run_kwargs):
        r"""
        Tests all loaded models and datasets, one dataset after the other.
        For each dataset test, the whole test set is iterated through and each
        data point is fed into each model to obtain performance metrics.
        After iteration, visualizations are created of the predictions of all models on randomly selected
        frame sequences, and performance metrics of each model on each dataset are logged in a comparable way.

        Args:
            brief_test (bool): If specified, only a brief sample check is done instead of fully iterating over the test set.
            **run_kwargs (Any): Optional testing configuration.
        """
        test_sets_and_model_lists, run_config = self._prepare_testing(**run_kwargs)
        for test_set, model_info_list in test_sets_and_model_lists:
            self._test_on_dataset(model_info_list, test_set, run_config, brief_test)
