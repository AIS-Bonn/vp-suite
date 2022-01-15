import os

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
from vp_suite.utils.utils import timestamp
from vp_suite.utils.compatibility import check_run_and_model_compat, check_model_and_data_compat
from vp_suite.utils.visualization import visualize_vid

class Tester(Runner):

    def __init__(self, device="cpu"):
        super(Tester, self).__init__(device)

    def _reset_datasets(self):
        self.test_data = None

    def _reset_models(self):
        self.models = []

    def _load_dataset(self, dataset_class, img_processor, **dataset_kwargs):
        self.test_data = dataset_class.get_test(img_processor, **dataset_kwargs)
        self.dataset = self.test_data

    def _prepare_testing(self, **run_kwargs):
        """
        Updates the current config with the given training parameters,
        prepares the dataset for usage and checks model compatibility.
        """

        run_config = self._get_run_config(**run_kwargs)

        # prepare dataset for testing
        self.test_data.set_seq_len(run_config["context_frames"], run_config["pred_frames"],
                                      run_config["seq_step"])

        # check model compatibility, receiving adapters if models can be made working with them
        for model_info in self.models:
            model, model_dir, _, _, _ = model_info
            check_model_and_data_compat(model, self.dataset_config)
            preprocessing, postprocessing, = check_run_and_model_compat(model, run_config, model_dir=model_dir)
            model_info[2] = preprocessing
            model_info[3] = postprocessing

        # all models OK -> finalize and add baseline copy model (doesn't need check)
        self.run_config = run_config
        clf_baseline_ = CopyLastFrame().to(self.device)
        clf_baseline = (clf_baseline_, None, nn.Identity(), nn.Identity(), [])
        self.models.append(clf_baseline)

    def load_models(self, model_dirs, ckpt_name="best_model.pth"):
        """
        overrides existing models
        desc: (model, model_dir, model_cfg, preprocessing, postprocessing, test_metrics (initialized empty))
        """
        self.reset_models()
        for model_dir in model_dirs:
            model = torch.load(os.path.join(model_dir, ckpt_name)).to(self.device)
            model_info = (model, model_dir, None, None, [])
            self.models.append(model_info)
        self.models_ready = True

    def test(self, brief_test=False, **run_kwargs):

        # PREPARATION
        self._prepare_testing(**run_kwargs)
        test_loader = DataLoader(self.test_data, batch_size=1, shuffle=True, num_workers=0)
        iter_loader = iter(test_loader)
        test_mode = "brief" if brief_test else "full"
        eval_length = 10 if test_mode == "brief" else len(iter_loader)

        # assemble and save combined configuration
        config = {**self.run_config, **self.dataset_config, "device": self.device}

        # evaluation / metric calc.
        if eval_length > 0:
            with torch.no_grad():

                metric_provider = PredictionMetricProvider(config)

                for _ in tqdm(range(eval_length)):
                    data = next(iter_loader)
                    img_data = data["frames"].to(config["device"])
                    input = img_data[:, :config["context_frames"]]
                    target = img_data[:, config["context_frames"]:config["context_frames"] + config["pred_frames"]]
                    actions = data["actions"].to(config["device"])

                    for (model, _, preprocess, postprocess, model_metrics_per_dp) in self.models:
                        input = preprocess(input)  # test format to model format
                        if getattr(model, "use_actions", False):
                            pred, _ = model(input, pred_length=config["pred_frames"], actions=actions)
                        else:
                            pred, _ = model(input, pred_length=config["pred_frames"])
                        pred = postprocess(pred)  # model format to test format
                        cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                        model_metrics_per_dp.append(cur_metrics)

        # save visualizations
        if not config["no_vis"]:
            print(f"Saving visualizations for trained models...")
            num_vis = 5
            vis_idx = np.random.choice(len(self.test_data), num_vis, replace=False)
            for i, (model, model_dir, model_cfg, _, _, _) in enumerate(self.models):
                if model_dir is None:
                    continue  # don't print for models that don't have a run dir (i.e. baseline models)
                vis_out_dir = Path(model_dir) / f"vis_{timestamp('test')}"
                vis_out_dir.mkdir()
                visualize_vid(self.test_data, config["context_frames"], config["pred_frames"],
                              model, config["device"], vis_out_dir, vis_idx=vis_idx)

        # log or display metrics
        if eval_length > 0:
            wandb_full_suffix = f"{test_mode} test"
            for i, (model, model_dir, _, _, model_metrics_per_dp) in enumerate(self.models):

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
                    wandb.init(config={"test_mode": test_mode, "model_dir": model_dir},
                               project="vp-suite-testing", name=f"{model.NAME}{wandb_full_suffix}",
                               dir=str(constants.WANDB_PATH.resolve()), reinit=(i > 0))
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        wandb.log({"pred_frames": f+1, **mean_metric_dict})
                    if i == len(self.models) - 1:
                        wandb.finish()

                # Per-model/per-pred.-horizon console log of the metrics
                else:
                    print("Printing out test results to terminal...")
                    print(f"\n{model.NAME} (path: {model_dir}): ")
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        print(f"pred_frames: {f + 1}")
                        for (k, v) in mean_metric_dict.items():
                            print(f" -> {k}: {v}")
