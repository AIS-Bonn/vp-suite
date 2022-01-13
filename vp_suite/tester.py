import sys, os, random, json
sys.path.append("")
from copy import deepcopy

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import vp_suite.constants as constants
from vp_suite.runner import Runner
from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.measure.metric_provider import PredictionMetricProvider
from vp_suite.utils.utils import timestamp, check_model_compatibility
from vp_suite.utils.visualization import visualize_vid
from vp_suite.dataset._factory import update_cfg_from_dataset, DATASET_CLASSES

class Tester(Runner):

    DEFAULT_TESTER_CONFIG = (constants.PKG_RESOURCES / 'run_config.json').resolve()

    def __init__(self, device="cpu"):
        super(Tester, self).__init__(device)

    def _reset_datasets(self):
        self.test_data = None

    def _reset_models(self):
        self.models_dict = {}

    def _load_dataset(self, dataset_class, **dataset_kwargs):
        self.test_data = dataset_class.get_test(self.img_processor, **dataset_kwargs)
        self.dataset = self.test_data

    def _prepare_testing(self, **testing_kwargs):
        """
        Updates the current config with the given training parameters,
        prepares the dataset for usage and checks model compatibility.
        """

        assert self.datasets_ready, "No datasets loaded. Load a dataset before starting testing"
        assert self.models_ready, "No models available. Load some trained models before starting testing"
        updated_config = deepcopy(self.config)  # TODO limit which args can be specified
        updated_config.update(testing_kwargs)

        # prepare dataset for testing
        self.test_data.set_seq_len(updated_config["context_frames"], updated_config["pred_frames"],
                                      updated_config["seq_step"])

        # check model compatibility, receiving adapters if models can be made working with them
        for model_desc, (model, model_dir, model_config, _, _, _) in self.models_dict.items():
            if model_config != self.config:
                preprocessing, postprocessing, = check_model_compatibility(model_config, updated_config, model,
                                                                           model_dir=model_dir)
                self.models_dict[model_desc] = (model, model_dir, model_config, preprocessing, postprocessing, [])

        # all models OK -> finalize and add baseline copy model (doesn't need check)
        self.config = updated_config
        clf_baseline = CopyLastFrame().to(self.config["device"])
        self.models_dict[clf_baseline.desc] = (clf_baseline, None, self.config, nn.Identity(), nn.Identity(), [])

    def load_models(self, model_dirs, ckpt_name="best_model.pth", cfg_name="run_cfg.json"):
        """
        overrides existing models
        desc: (model, model_dir, model_cfg, preprocessing, postprocessing, test_metrics (initialized empty))
        """
        self.models_dict = {}
        for model_dir in model_dirs:
            model = torch.load(os.path.join(model_dir, ckpt_name)).to(self.config["device"])
            with open(os.path.join(model_dir, cfg_name), "r") as cfg_file:
                model_config = json.load(cfg_file)
            self.models_dict[model.desc] = (model, model_dir, model_config, None, None, [])
        self.models_ready = True

    def test(self, **testing_kwargs):

        # PREPARATION
        self._prepare_testing(**testing_kwargs)
        test_loader = DataLoader(self.test_data, batch_size=1, shuffle=True, num_workers=0)
        iter_loader = iter(test_loader)
        mini_test = testing_kwargs.get("mini", False)
        eval_length = 10 if mini_test else len(iter_loader)

        # evaluation / metric calc.
        if eval_length > 0:
            with torch.no_grad():

                metric_provider = PredictionMetricProvider(self.config)

                for _ in tqdm(range(eval_length)):
                    data = next(iter_loader)
                    img_data = data["frames"].to(self.config["device"])
                    input = img_data[:, :self.config["context_frames"]]
                    target = img_data[:, self.config["context_frames"]:self.config["context_frames"] + self.config["pred_frames"]]
                    actions = data["actions"].to(self.config["device"])

                    for (model, _, _, preprocess, postprocess, model_metrics_per_dp) in self.models_dict.values():
                        input = preprocess(input)  # test format to model format
                        if getattr(model, "use_actions", False):
                            pred, _ = model.pred_n(input, pred_length=self.config["pred_frames"], actions=actions)
                        else:
                            pred, _ = model.pred_n(input, pred_length=self.config["pred_frames"])
                        pred = postprocess(pred)  # model format to test format
                        cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                        model_metrics_per_dp.append(cur_metrics)

        # save visualizations
        if not self.config["no_vis"]:
            print(f"Saving visualizations for trained models...")
            num_vis = 5
            vis_idx = np.random.choice(len(self.test_data), num_vis, replace=False)
            for i, (model_desc, (model, model_dir, model_cfg, _, _, _)) in enumerate(self.models_dict.items()):
                if model_dir is None:
                    continue  # don't print for models that don't have a run dir (i.e. baseline models)
                vis_out_dir = Path(model_dir) / f"vis_{timestamp('test')}"
                vis_out_dir.mkdir()
                visualize_vid(self.test_data, self.config["context_frames"], self.config["pred_frames"], model, self.config["device"],
                              self.img_processor, vis_out_dir, vis_idx=vis_idx)

        # log or display metrics
        if eval_length > 0:
            wandb_full_suffix = "" if mini_test else "(full test)"
            models_dict_items = self.models_dict.items()
            for i, (model_desc, (_, model_dir, model_cfg, _, _, model_metrics_per_dp)) in enumerate(models_dict_items):

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
                if not self.config["no_wandb"]:
                    print("Logging test results to WandB for all models...")
                    wandb.init(config={"mini_test": mini_test, "model_dir": model_dir},
                               project="vp-suite-testing", name=f"{model_desc}{wandb_full_suffix}",
                               dir=str(constants.WANDB_PATH.resolve()), reinit=(i > 0))
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        wandb.log({"pred_frames": f+1, **mean_metric_dict})
                    if i == len(models_dict_items) - 1:
                        wandb.finish()

                # Per-model/per-pred.-horizon console log of the metrics
                else:
                    print("Printing out test results to terminal...")
                    print(f"\n{model_desc} (path: {model_dir}): ")
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        print(f"pred_frames: {f + 1}")
                        for (k, v) in mean_metric_dict.items():
                            print(f" -> {k}: {v}")
