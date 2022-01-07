import json
import os
import random
from pathlib import Path
from copy import deepcopy

import wandb

import numpy as np
import torch.nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import vp_suite.constants as constants
from vp_suite.dataset._factory import update_cfg_from_dataset, DATASET_CLASSES
from vp_suite.models._factory import create_pred_model
from vp_suite.utils.img_processor import ImgProcessor
from vp_suite.measure.loss_provider import PredictionLossProvider, LOSSES
from vp_suite.utils.visualization import visualize_vid
from vp_suite.utils.utils import timestamp, check_model_compatibility

class Trainer:

    DEFAULT_TRAINER_CONFIG = (constants.PKG_RESOURCES / 'run_config.json').resolve()

    def __init__(self):
        with open(self.DEFAULT_TRAINER_CONFIG, 'r') as tc_file:
            self.config = json.load(tc_file)
        self.config["opt_direction"] = "maximize" if LOSSES[self.config["val_rec_criterion"]].bigger_is_better \
            else "minimize"
        self.config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_processor = ImgProcessor(self.config["tensor_value_range"])
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])

        self.datasets_loaded = False
        self.pred_model = None
        self.model_config = None
        self.model_ready = False

    def load_dataset(self, dataset="MM", **dataset_kwargs):
        """
        ATTENTION: this removes any loaded models
        """
        self.pred_model = None
        self.model_config = None
        self.model_ready = False
        dataset_class = DATASET_CLASSES[dataset]
        self.train_data, self.val_data = dataset_class.get_train_val(self.img_processor, **dataset_kwargs)
        dataset = self.train_data.dataset if isinstance(self.train_data, Subset) else self.train_data
        self.config = update_cfg_from_dataset(self.config, self.train_data)
        print(f"INFO: loaded dataset '{dataset.NAME}' from {dataset.data_dir} (action size: {dataset.ACTION_SIZE})")
        self.datasets_loaded = True

    def load_model(self, model_dir, ckpt_name="best_model.pth", cfg_name="run_cfg.json"):
        """
        overrides existing model
        """
        model_ckpt = os.path.join(model_dir, ckpt_name)
        loaded_model = torch.load(model_ckpt)
        with open(os.path.join(model_dir, cfg_name), "r") as cfg_file:
            model_config = json.load(cfg_file)
        _, _, = check_model_compatibility(model_config, self.config, loaded_model,
                                          strict_mode=True, model_dir=model_dir)
        self.pred_model = loaded_model
        self.model_config = model_config
        print(f"INFO: loaded pre-trained model '{self.pred_model.desc}' from {model_ckpt}")
        self.model_ready = True

    def create_model(self, model_type, **model_args):
        """
        overrides existing model
        """
        self.pred_model = create_pred_model(self.config, model_type, **model_args)
        self.model_config = self.config
        ac_str = "(action-conditional)" if self.config["use_actions"] and self.pred_model.can_handle_actions else ""
        print(f"INFO: created new model '{self.pred_model.desc}' {ac_str}")
        total_params = sum(p.numel() for p in self.pred_model.parameters())
        trainable_params = sum(p.numel() for p in self.pred_model.parameters() if p.requires_grad)
        print(f" - Model parameters (total / trainable): {total_params} / {trainable_params}")
        self.model_ready = True

    def _prepare_training(self, **training_kwargs):
        """
        Updates the current config with the given training parameters,
        prepares the dataset for usage and checks model compatibility.
        """

        assert self.datasets_loaded, "No datasets loaded. Load a dataset before starting training"
        assert self.model_ready, "No model available. Load a pretrained model or create a new instance before starting training"
        updated_config = deepcopy(self.config)  # TODO limit which args can be specified
        updated_config.update(training_kwargs)

        # prepare datasets for training
        if isinstance(self.train_data, Subset):
            self.train_data.dataset.set_seq_len(updated_config["context_frames"], updated_config["pred_frames"],
                                    updated_config["seq_step"])
        else:
            self.train_data.set_seq_len(updated_config["context_frames"], updated_config["pred_frames"],
                                        updated_config["seq_step"])
            self.val_data.set_seq_len(updated_config["context_frames"], updated_config["pred_frames"],
                                      updated_config["seq_step"])

        # check model compatibility
        if self.model_config != updated_config:
            _, _, = check_model_compatibility(self.model_config, updated_config, self.pred_model, strict_mode=True)

        self.config = updated_config

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

    def hyperopt(self, optuna_cfg=None, n_trials=30, **training_kwargs):
        self._set_optuna_cfg(optuna_cfg)
        from functools import partial
        try:
            import optuna
        except ImportError:
            raise ImportError("Importing optuna failed -> install it or use the code without the 'use-optuna' flag.")
        optuna_program = partial(self.train, **training_kwargs)
        study = optuna.create_study(direction=self.config["opt_direction"])
        study.optimize(optuna_program, n_trials=n_trials)
        print("\nHyperparameter optimization complete. Best performing parameters:")
        for k, v in study.best_params.items():
            print(f" - {k}: {v}")

    def train(self, trial=None, **training_kwargs):

        # PREPARATION
        self._prepare_training(**training_kwargs)
        train_loader = DataLoader(self.train_data, batch_size=self.config["batch_size"], shuffle=True, num_workers=4,
                                  drop_last=True)
        val_loader = DataLoader(self.val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        best_val_loss = float("inf")
        out_path = constants.OUT_PATH / timestamp('train')
        out_path.mkdir(parents=True)
        best_model_path = str((out_path / 'best_model.pth').resolve())
        with_training = self.pred_model.trainable and not self.config["no_train"]

        if self.config["opt_direction"] == "maximize":
            def loss_improved(cur_loss, best_loss): return cur_loss > best_loss
        else:
            def loss_improved(cur_loss, best_loss): return cur_loss < best_loss

        # HYPERPARAMETER OPTIMIZATION
        using_optuna = trial is not None
        if using_optuna:
            assert self.optuna_cfg is not None, "optuna_cfg is None -> can't hyperopt"
            for param, p_dict in self.optuna_cfg.items():
                if "choices" in p_dict.keys():
                    if param == "model_type":
                        print(f"WARNING: hyperopt across different model types is not yet supported "
                              f"-> using {self.pred_model.desc}")
                    self.config[param] = trial.suggest_categorical(param, p_dict["choices"])
                else:
                    suggest = trial.suggest_int if p_dict["type"] == "int" else trial.suggest_float
                    log_scale = p_dict.get("scale", "uniform") == "log"
                    if log_scale:
                        self.config[param] = suggest(param, p_dict["min"], p_dict["max"], log=log_scale)
                    else:
                        step = p_dict.get("step", 1)
                        self.config[param] = suggest(param, p_dict["min"], p_dict["max"], step=step)

        # WandB
        if not self.config["no_wandb"]:
            wandb_reinit = using_optuna and trial.number > 0
            wandb.init(config=self.config, project="vp-suite-training", reinit=wandb_reinit)

        # OPTIMIZER
        optimizer, optimizer_scheduler = None, None
        if with_training:
            optimizer = torch.optim.Adam(params=self.pred_model.parameters(), lr=self.config["lr"])
            optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2,
                                                                             min_lr=1e-6, verbose=True)

        # LOSSES
        loss_provider = PredictionLossProvider(self.config)
        if self.config['val_rec_criterion'] not in self.config['losses_and_scales']:
            raise ValueError(f"ERROR: Validation criterion '{self.config['val_rec_criterion']}' has to be "
                             f"one of the chosen losses: {list(self.config['losses_and_scales'].keys())}")

        # save run config
        with open(str((out_path / 'run_cfg.json').resolve()), "w") as cfg_file:
            json.dump(self.config, cfg_file, indent=4,
                      default=lambda o: str(o) if callable(getattr(o, "__str__", None)) else '<not serializable>')

        # --- MAIN LOOP ---
        for epoch in range(0, self.config["epochs"]):

            # train
            print(f'\nTraining (epoch: {epoch+1} of {self.config["epochs"]})')
            if with_training:
                # use prediction model's own training loop if available
                if callable(getattr(self.pred_model, "train_iter", None)):
                    self.pred_model.train_iter(self.config, train_loader, optimizer, loss_provider, epoch)
                else:
                    self.train_iter(train_loader, optimizer, loss_provider)
            else:
                print("Skipping training loop.")

            # eval
            print("Validating...")
            val_losses, indicator_loss = self.eval_iter(val_loader, loss_provider)
            if with_training:
                optimizer_scheduler.step(indicator_loss)
            print("Validation losses (mean over entire validation set):")
            for k, v in val_losses.items():
                print(f" - {k}: {v}")

            # save model if last epoch improved indicator loss
            cur_val_loss = indicator_loss.item()
            if loss_improved(cur_val_loss, best_val_loss):
                best_val_loss = cur_val_loss
                torch.save(self.pred_model, best_model_path)
                print(f"Minimum indicator loss ({self.config['val_rec_criterion']}) reduced -> model saved!")

            # visualize current model performance every nth epoch, using eval mode and validation data.
            if (epoch+1) % self.config["vis_every"] == 0 and not self.config["no_vis"]:
                print("Saving visualizations...")
                vis_out_path = out_path / f"vis_ep_{epoch+1:03d}"
                vis_out_path.mkdir()
                visualize_vid(self.val_data, self.config["context_frames"], self.config["pred_frames"], self.pred_model,
                              self.config["device"], self.img_processor, vis_out_path, num_vis=10)

                if not self.config["no_wandb"]:
                    vid_filenames = sorted(os.listdir(str(vis_out_path)))
                    log_vids = {fn: wandb.Video(str(vis_out_path / fn), fps=4, format=fn.split(".")[-1])
                                for i, fn in enumerate(vid_filenames)}
                    wandb.log(log_vids, commit=False)

            # final bookkeeping
            if not self.config["no_wandb"]:
                wandb.log(val_losses, commit=True)

        # finishing
        print("\nTraining done, cleaning up...")
        torch.save(self.pred_model, str((out_path / 'final_model.pth').resolve()))
        wandb.finish()
        return best_val_loss  # return best validation loss for hyperparameter optimization

    def train_iter(self, loader, optimizer, loss_provider):
        loop = tqdm(loader)
        for batch_idx, data in enumerate(loop):

            # input
            img_data = data["frames"].to(self.config["device"])  # [b, T, c, h, w], with T = total_frames
            input = img_data[:, :self.config["context_frames"]]
            targets = img_data[:, self.config["context_frames"]
                                  : self.config["context_frames"] + self.config["pred_frames"]]
            actions = data["actions"].to(self.config["device"])  # [b, T-1, a]. Action t corresponds to what happens after frame t

            # fwd
            predictions, model_losses = self.pred_model.pred_n(input, pred_length=self.config["pred_frames"],
                                                               actions=actions)

            # loss
            _, total_loss = loss_provider.get_losses(predictions, targets)
            if model_losses is not None:
                for value in model_losses.values():
                    total_loss += value

            # bwd
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), 100)
            optimizer.step()

            # bookkeeping
            loop.set_postfix(loss=total_loss.item())
            # loop.set_postfix(mem=torch.cuda.memory_allocated())

    def eval_iter(self, loader, loss_provider):
        self.pred_model.eval()
        loop = tqdm(loader)
        all_losses = []
        indicator_losses = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loop):

                # fwd
                img_data = data["frames"].to(self.config["device"])  # [b, T, h, w], with T = total_frames
                input = img_data[:, :self.config["context_frames"]]
                targets = img_data[:, self.config["context_frames"]
                                      : self.config["context_frames"] + self.config["pred_frames"]]
                actions = data["actions"].to(self.config["device"])

                predictions, model_losses = self.pred_model.pred_n(input, pred_length=self.config["pred_frames"],
                                                                   actions=actions)

                # metrics
                loss_values, _ = loss_provider.get_losses(predictions, targets)
                all_losses.append(loss_values)
                indicator_losses.append(loss_values[self.config["val_rec_criterion"]])

        indicator_loss = torch.stack(indicator_losses).mean()
        all_losses = {
            k: torch.stack([loss_values[k] for loss_values in all_losses]).mean().item() for k in all_losses[0].keys()
        }
        self.pred_model.train()

        return all_losses, indicator_loss