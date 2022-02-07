import random, json, os
import warnings
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

import vp_suite.constants as constants
from vp_suite.utils.dataset_wrapper import DatasetWrapper
from vp_suite.datasets import DATASET_CLASSES
from vp_suite.base.base_model import VideoPredictionModel
from vp_suite.models import MODEL_CLASSES, AVAILABLE_MODELS
from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.measure import LOSS_CLASSES
from vp_suite.measure.loss_provider import PredictionLossProvider
from vp_suite.measure.metric_provider import PredictionMetricProvider
from vp_suite.utils.visualization import visualize_vid
from vp_suite.utils.utils import timestamp
from vp_suite.utils.compatibility import check_model_and_data_compat, check_run_and_model_compat


class VPSuite:
    r"""TODO.

    TODO.

    Attributes:
        device(str): A string specifying which device to work on ('cuda' or 'gpu')
        datasets(List[:class:`DatasetWrapper`]): The loaded datasets.
        models(List[:class:`VideoPredictionModel`]): The loaded/created models.

    """

    _DEFAULT_RUN_CONFIG : Path = constants.PKG_RESOURCES / 'run_config.json'

    def __init__(self, device="cuda"):
        r"""

        Args:
            device ():
        """
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.clear_models()
        self.clear_datasets()

    @property
    def training_sets(self):
        r"""

        Returns:

        """
        return [d for d in self.datasets if d.is_training_set()]

    @property
    def test_sets(self):
        r"""

        Returns:

        """
        return [d for d in self.datasets if d.is_test_set()]

    def clear_datasets(self):
        r"""

        Returns:

        """
        self.datasets : List[DatasetWrapper] = []

    def clear_models(self):
        r"""

        Returns:

        """
        self.models : List[VideoPredictionModel] = []

    def load_dataset(self, dataset, split="train", **dataset_kwargs):
        r"""

        Args:
            dataset ():
            split ():
            value_min ():
            value_max ():
            **dataset_kwargs ():

        Returns:

        """
        dataset_class = DATASET_CLASSES[dataset]
        dataset = DatasetWrapper(dataset_class, split, **dataset_kwargs)
        print(f"loaded dataset '{dataset.NAME}' from {dataset.data_dir} "
              f"(action size: {dataset.action_size})")
        self.datasets.append(dataset)

    def download_dataset(self, dataset):
        dataset_class = DATASET_CLASSES[dataset]
        dataset_class.download_and_prepare_dataset()

    def load_model(self, model_dir, ckpt_name="best_model.pth"):
        r"""

        Args:
            model_dir ():
            ckpt_name ():

        Returns:

        """
        model_ckpt = os.path.join(model_dir, ckpt_name)
        model = torch.load(model_ckpt)
        self._model_setup(model, loaded=True)

    def create_model(self, model_type, action_conditional=False, **model_kwargs):
        r"""

        Args:
            model_type ():
            action_conditional ():
            **model_kwargs ():

        Returns:

        """

        # parameter processing
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(f"invalid model type specified! Available model types: {AVAILABLE_MODELS}")

        model_class = MODEL_CLASSES[model_type]
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

    def _model_setup(self, model, loaded=False):
        r"""

        Args:
            model ():
            loaded ():

        Returns:

        """
        ac_str = "(action-conditional)" if model.config["action_conditional"] else ""
        loaded_str = "loaded" if loaded else "created new"
        print(f"{loaded_str} model '{model.NAME}' {ac_str}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" - Model parameters (total / trainable): {total_params} / {trainable_params}")
        self.models.append(model)

    def _prepare_run(self, split="train", **run_args):
        r"""

        Args:
            split ():
            **run_args ():

        Returns:

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
        with open(str(self._DEFAULT_RUN_CONFIG.resolve()), 'r') as tc_file:
            run_config = json.load(tc_file)

        # update config
        if not all([run_arg in run_config.keys() for run_arg in run_args.keys()]):
            raise ValueError(f"Only the following run arguments are supported: {run_config.keys()}")
        run_config.update(run_args)

        # seed
        random.seed(run_config["seed"])
        np.random.seed(run_config["seed"])
        torch.manual_seed(run_config["seed"])

        # opt. direction
        run_config["opt_direction"] = "maximize" if LOSS_CLASSES[run_config["val_rec_criterion"]].BIGGER_IS_BETTER \
            else "minimize"

        return run_config

# ===== TRAINING ================================================================

    def _prepare_training(self, dataset_idx, model_idx, **training_kwargs):
        r"""
        Prepares a single dataset and a single model for training.
        Args:
            **training_kwargs ():

        Returns:

        """
        run_config = self._prepare_run("train", **training_kwargs)

        try:
            dataset : DatasetWrapper = self.training_sets[dataset_idx]
            model : VideoPredictionModel = self.models[model_idx]
        except IndexError:
            raise ValueError("given indices for model and/or dataset are invalid")

        # prepare dataset
        dataset.set_seq_len(run_config["context_frames"], run_config["pred_frames"], run_config["seq_step"])
        assert dataset.is_ready, "dataset is not ready even though set_seq_len has just been called"

        # compat checks: run <--> model; model <--> dataset
        check_run_and_model_compat(model, run_config)
        _, _ = check_model_and_data_compat(model, dataset, strict_mode=True)

        return model, dataset, run_config

    def train(self, trial=None, dataset_idx=-1, model_idx=-1, **training_kwargs):
        r"""

        Args:
            trial ():
            **training_kwargs ():

        Returns:

        """
        # PREPARATION
        model, dataset, run_config = self._prepare_training(dataset_idx, model_idx, **training_kwargs)
        train_data, val_data = dataset.train_data, dataset.val_data
        train_loader = DataLoader(train_data, batch_size=run_config["batch_size"], shuffle=True, num_workers=4,
                                  drop_last=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        best_val_loss = float("inf")
        out_path = Path(run_config["out_dir"]) if run_config["out_dir"] is not None \
            else constants.OUT_PATH / timestamp('train')
        out_path.mkdir(parents=True, exist_ok=True)
        model.model_dir = str(out_path.resolve())
        best_model_path = str((out_path / 'best_model.pth').resolve())
        with_training = model.TRAINABLE and not run_config["no_train"]

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
        config : Dict[str, Any] = {**run_config, **model.config, **dataset.config, "device": self.device}
        save_config = {"run": run_config, "model": model.config,
                       "dataset": dataset.config, "device": self.device}
        with open(str((out_path / 'run_cfg.json').resolve()), "w") as cfg_file:
            json.dump(save_config, cfg_file, indent=4,
                      default=lambda o: str(o) if callable(getattr(o, "__str__", None)) else '<not serializable>')

        # WandB
        if not run_config["no_wandb"]:
            wandb_reinit = using_optuna and trial.number > 0
            wandb.init(config=config, project="vp-suite-training",
                       dir=str(constants.WANDB_PATH.resolve()), reinit=wandb_reinit)

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
        for epoch in range(0, run_config["epochs"]):

            # train
            print(f'\nTraining (epoch: {epoch+1} of {config["epochs"]})')
            if with_training:
                model.train_iter(config, train_loader, optimizer, loss_provider, epoch)
            else:
                print("Skipping training loop.")

            # eval
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

            # visualize current model performance every nth epoch, using eval mode and validation data.
            if (epoch+1) % config["vis_every"] == 0 and not config["no_vis"]:
                print("Saving visualizations...")
                vis_out_path = out_path / f"vis_ep_{epoch+1:03d}"
                vis_out_path.mkdir()
                visualize_vid(val_data, config["context_frames"], config["pred_frames"], model,
                              config["device"], vis_out_path, num_vis=10)

                if not config["no_wandb"]:
                    vid_filenames = sorted(os.listdir(str(vis_out_path)))
                    log_vids = {fn: wandb.Video(str(vis_out_path / fn), fps=4, format=fn.split(".")[-1])
                                for i, fn in enumerate(vid_filenames)}
                    wandb.log(log_vids, commit=False)

            # final bookkeeping
            if not config["no_wandb"]:
                wandb.log(val_losses, commit=True)

        # finishing
        print("\nTraining done, cleaning up...")
        torch.save(model, str((out_path / 'final_model.pth').resolve()))
        wandb.finish()
        return best_val_loss  # return best validation loss for hyperparameter optimization

    def hyperopt(self, optuna_config, n_trials=30, dataset_idx=-1, model_idx=-1, **run_kwargs):
        r"""

        Args:
            optuna_config ():
            n_trials ():
            **run_kwargs ():

        Returns:

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
        Prepares multiple datasets and models for testing
        Args:
            **run_kwargs ():

        Returns:

        """
        run_config = self._prepare_run("test", **run_kwargs)

        # prepare datasets for testing
        test_sets : List[DatasetWrapper] = self.test_sets
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

    def _test_on_dataset(self, model_info_list, dataset, run_config, brief_test):
        r"""

        Args:
            model_info_list ():
            dataset ():
            run_config ():
            brief_test ():

        Returns:

        """
        # PREPARATION
        test_data = dataset.test_data
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
        if len(test_loader) < 1:
            raise RuntimeError("loaded dataset does not contain any data (len < 1)")
        test_mode = "brief" if brief_test else "full"
        eval_length = min(len(test_loader), 10) if test_mode == "brief" else len(test_loader)

        # assemble and save combined configuration
        config = {**run_config, **dataset.config, "device": self.device}

        # evaluation / metric calc.
        iter_loader = iter(test_loader)
        with torch.no_grad():
            metric_provider = PredictionMetricProvider(config)

            for _ in tqdm(range(eval_length)):
                data = next(iter_loader)

                for (model, preprocess, postprocess, model_metrics_per_dp) in model_info_list:
                    input, target, actions = model.unpack_data(data, config)
                    input = preprocess(input)  # test format to model format
                    if getattr(model, "use_actions", False):
                        pred, _ = model(input, pred_frames=config["pred_frames"], actions=actions)
                    else:
                        pred, _ = model(input, pred_frames=config["pred_frames"])
                    pred = postprocess(pred)  # model format to test format
                    cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                    model_metrics_per_dp.append(cur_metrics)

        # save visualizations
        if not config["no_vis"]:
            print(f"Saving visualizations for trained models...")
            num_vis = 5
            vis_idx = np.random.choice(len(test_data), num_vis, replace=False)
            for i, (model, _, _, _) in enumerate(model_info_list):
                if model.model_dir is None:
                    continue  # don't print for models that don't have a run dir (i.e. baseline models)
                vis_out_dir = Path(model.model_dir) / f"vis_{timestamp('test')}"
                vis_out_dir.mkdir()
                visualize_vid(test_data, config["context_frames"], config["pred_frames"],
                              model, config["device"], vis_out_dir, vis_idx=vis_idx)

        # log or display metrics
        if eval_length > 0:
            wandb_full_suffix = f"{test_mode} test"
            for i, (model, _, _, model_metrics_per_dp) in enumerate(model_info_list):
                # model_metrics_per_dp: list of N lists of F metric dicts (called D).
                # each D_{n, f} contains all calculated metrics for datapoint 'n' and a prediction horizon of 'f' frames.
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
                if not config["no_wandb"]:
                    print("Logging test results to WandB for all models...")
                    wandb.init(config={"test_mode": test_mode, "model_dir": model.model_dir},
                               project="vp-suite-testing", name=f"{model.NAME}{wandb_full_suffix}",
                               dir=str(constants.WANDB_PATH.resolve()), reinit=(i > 0))
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        wandb.log({"pred_frames": f+1, **mean_metric_dict})
                    if i == len(model_info_list) - 1:
                        wandb.finish()

                # Per-model/per-pred.-horizon console log of the metrics
                else:
                    print("Printing out test results to terminal...")
                    print(f"\n{model.NAME} (path: {model.model_dir}): ")
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        print(f"pred_frames: {f + 1}")
                        for (k, v) in mean_metric_dict.items():
                            print(f" -> {k}: {v}")

    def test(self, brief_test=False, **run_kwargs):
        r"""

        Args:
            brief_test ():
            **run_kwargs ():

        Returns:

        """
        test_sets_and_model_lists, run_config = self._prepare_testing(**run_kwargs)
        for test_set, model_info_list in test_sets_and_model_lists:
            self._test_on_dataset(model_info_list, test_set, run_config, brief_test)
