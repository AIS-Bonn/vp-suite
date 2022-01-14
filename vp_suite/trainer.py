import json
import os

import wandb

import torch.nn
from torch.utils.data import DataLoader, Subset

import vp_suite.constants as constants
from vp_suite.runner import Runner
from vp_suite.models import MODEL_CLASSES, AVAILABLE_MODELS
from vp_suite.measure.loss_provider import PredictionLossProvider
from vp_suite.utils.visualization import visualize_vid
from vp_suite.utils.utils import timestamp
from vp_suite.utils.compatibility import check_model_and_data_compat, check_run_and_model_compat


class Trainer(Runner):

    def __init__(self, device="cpu"):
        super(Trainer, self).__init__(device)

    @property
    def model_config(self):
        if self.model is None:
            return None
        elif self.is_loaded_model:
            return self.loaded_model_config
        else:
            return self.model.config

    def _reset_datasets(self):
        self.train_data = None
        self.val_data = None

    def _reset_models(self):
        """ removes any loaded models """
        self.model = None
        self.is_loaded_model = False
        self.loaded_model_config = None

    def _load_dataset(self, dataset_class, img_processor, **dataset_kwargs):
        self.train_data, self.val_data = dataset_class.get_train_val(img_processor, **dataset_kwargs)
        self.dataset = self.train_data.dataset if isinstance(self.train_data, Subset) else self.train_data

    def load_model(self, model_dir, ckpt_name="best_model.pth"):
        """
        overrides existing model
        """
        self.reset_models()
        assert self.datasets_ready, "ERROR: no dataset loaded -> load a dataset first before loading a model."

        model_ckpt = os.path.join(model_dir, ckpt_name)
        model = torch.load(model_ckpt)
        self._model_setup(model, loaded=True)

    def create_model(self, model_type, action_conditional=False, **model_args):
        """
        overrides existing model
        """
        self.reset_models()
        assert self.datasets_ready, "ERROR: no dataset loaded -> load a dataset first before loading a model."

        # parameter processing
        assert model_type in AVAILABLE_MODELS, f"ERROR: invalid model type specified! " \
                                               f"Available model types: {AVAILABLE_MODELS}"
        if action_conditional and not self.model.CAN_HANDLE_ACTIONS:
            print("WARNING: specified model can't handle actions -> argument 'action_conditional' set to False")
            action_conditional = False
        model_args.update(action_conditional=action_conditional)

        # model creation
        model = MODEL_CLASSES[model_type](self.dataset_config, self.device, **model_args).to(self.device)
        self._model_setup(model)

    def _model_setup(self, model, loaded=False):
        """ TODO docs """
        check_model_and_data_compat(model, self.dataset_config)
        self.model = model
        ac_str = "(action-conditional)" if self.model_config["action_conditional"] else ""
        loaded_str = "loaded" if loaded else "created new"
        print(f"INFO: {loaded_str} model '{self.model.NAME}' {ac_str}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f" - Model parameters (total / trainable): {total_params} / {trainable_params}")
        self.models_ready = True

    def _prepare_training(self, **training_kwargs):
        """
        Updates the current config with the given training parameters,
        prepares the dataset for usage and checks model compatibility.
        """
        run_config = self._get_run_config(**training_kwargs)

        # prepare datasets and check compatibility
        train_data = self.train_data.dataset if isinstance(self.train_data, Subset) else self.train_data
        train_data.set_seq_len(run_config["context_frames"], run_config["pred_frames"],
                               run_config["seq_step"])
        self.val_data.set_seq_len(run_config["context_frames"], run_config["pred_frames"],
                                  run_config["seq_step"])

        # Compatibility needs to be checked with the loaded dataset and the run parameters
        _, _, = check_run_and_model_compat(self.model, run_config, strict_mode=True)
        check_model_and_data_compat(self.model, self.dataset_config)

        self.run_config = run_config

    def _set_optuna_cfg(self, optuna_cfg : dict):
        self.optuna_cfg = optuna_cfg if optuna_cfg else {}
        for parameter, p_dict in self.optuna_cfg.items():
            assert isinstance(p_dict, dict)
            if "choices" in p_dict.keys():
                assert(isinstance(p_dict["choices"], list))
            else:
                assert {"type", "min", "max"}.issubset(set(p_dict.keys()))
                assert p_dict["min"] <= p_dict["max"]
                if p_dict["type"] == "float":
                    assert p_dict.get("scale", '') in ["log", "uniform"]

    def hyperopt(self, optuna_cfg=None, n_trials=30, **run_kwargs):

        run_config = self._get_run_config(**run_kwargs)
        self._set_optuna_cfg(optuna_cfg)
        from functools import partial
        try:
            import optuna
        except ImportError:
            raise ImportError("Importing optuna failed -> install it or use the code without the 'use-optuna' flag.")
        optuna_program = partial(self.train, **run_kwargs)
        study = optuna.create_study(direction=run_config["opt_direction"])
        study.optimize(optuna_program, n_trials=n_trials)
        print("\nHyperparameter optimization complete. Best performing parameters:")
        for k, v in study.best_params.items():
            print(f" - {k}: {v}")

    def train(self, trial=None, **training_kwargs):

        # PREPARATION
        self._prepare_training(**training_kwargs)
        train_loader = DataLoader(self.train_data, batch_size=self.run_config["batch_size"], shuffle=True, num_workers=4,
                                  drop_last=True)
        val_loader = DataLoader(self.val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        best_val_loss = float("inf")
        out_path = constants.OUT_PATH / timestamp('train')
        out_path.mkdir(parents=True)
        best_model_path = str((out_path / 'best_model.pth').resolve())
        with_training = self.model.TRAINABLE and not self.run_config["no_train"]

        # HYPERPARAMETER OPTIMIZATION
        using_optuna = trial is not None
        if using_optuna:
            assert self.optuna_cfg is not None, "optuna_cfg is None -> can't hyperopt"
            for param, p_dict in self.optuna_cfg.items():
                if "choices" in p_dict.keys():
                    if param == "model_type":
                        print(f"WARNING: hyperopt across model and dataset parameters is not yet supported "
                              f"-> using {self.model.NAME}")
                    self.run_config[param] = trial.suggest_categorical(param, p_dict["choices"])
                else:
                    suggest = trial.suggest_int if p_dict["type"] == "int" else trial.suggest_float
                    log_scale = p_dict.get("scale", "uniform") == "log"
                    if log_scale:
                        self.run_config[param] = suggest(param, p_dict["min"], p_dict["max"], log=log_scale)
                    else:
                        step = p_dict.get("step", 1)
                        self.run_config[param] = suggest(param, p_dict["min"], p_dict["max"], step=step)

        # assemble and save combined configuration
        config = {**self.run_config, **self.model_config, **self.dataset_config, "device": self.device}
        save_config = {"run": self.run_config, "model": self.model_config,
                       "dataset": self.dataset_config, "device": self.device}
        with open(str((out_path / 'run_cfg.json').resolve()), "w") as cfg_file:
            json.dump(save_config, cfg_file, indent=4,
                      default=lambda o: str(o) if callable(getattr(o, "__str__", None)) else '<not serializable>')

        # WandB
        if not self.run_config["no_wandb"]:
            wandb_reinit = using_optuna and trial.number > 0
            wandb.init(config=config, project="vp-suite-training",
                       dir=str(constants.WANDB_PATH.resolve()), reinit=wandb_reinit)

        # OPTIMIZER
        optimizer, optimizer_scheduler = None, None
        if with_training:
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.run_config["lr"])
            optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                             min_lr=1e-6, verbose=True)

        # LOSSES AND MEASUREMENT
        loss_provider = PredictionLossProvider(config)
        if config['val_rec_criterion'] not in config['losses_and_scales']:
            raise ValueError(f"ERROR: Validation criterion '{config['val_rec_criterion']}' has to be "
                             f"one of the chosen losses: {list(config['losses_and_scales'].keys())}")
        if config["opt_direction"] == "maximize":
            def loss_improved(cur_loss, best_loss): return cur_loss > best_loss
        else:
            def loss_improved(cur_loss, best_loss): return cur_loss < best_loss

        # --- MAIN LOOP ---
        for epoch in range(0, config["epochs"]):

            # train
            print(f'\nTraining (epoch: {epoch+1} of {config["epochs"]})')
            if with_training:
                self.model.train_iter(config, train_loader, optimizer, loss_provider, epoch)
            else:
                print("Skipping training loop.")

            # eval
            print("Validating...")
            val_losses, indicator_loss = self.model.eval_iter(config, val_loader, loss_provider)
            if with_training:
                optimizer_scheduler.step(indicator_loss)
            print("Validation losses (mean over entire validation set):")
            for k, v in val_losses.items():
                print(f" - {k}: {v}")

            # save model if last epoch improved indicator loss
            cur_val_loss = indicator_loss.item()
            if loss_improved(cur_val_loss, best_val_loss):
                best_val_loss = cur_val_loss
                torch.save(self.model, best_model_path)
                print(f"Minimum indicator loss ({config['val_rec_criterion']}) reduced -> model saved!")

            # visualize current model performance every nth epoch, using eval mode and validation data.
            if (epoch+1) % config["vis_every"] == 0 and not config["no_vis"]:
                print("Saving visualizations...")
                vis_out_path = out_path / f"vis_ep_{epoch+1:03d}"
                vis_out_path.mkdir()
                visualize_vid(self.val_data, config["context_frames"], config["pred_frames"], self.model,
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
        torch.save(self.model, str((out_path / 'final_model.pth').resolve()))
        wandb.finish()
        return best_val_loss  # return best validation loss for hyperparameter optimization
