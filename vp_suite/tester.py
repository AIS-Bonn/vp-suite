import sys, os, random, json
sys.path.append("")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.measure.metric_provider import PredictionMetricProvider
from vp_suite.utils.utils import timestamp, check_model_compatibility
from vp_suite.utils.visualization import visualize_vid
from vp_suite.utils.img_processor import ImgProcessor
from vp_suite.dataset._factory import update_cfg_from_dataset, DATASET_CLASSES

class Tester:

    DEFAULT_TESTER_CONFIG = 'vp_suite/config.json'

    def __init__(self):
        with open(self.DEFAULT_TESTER_CONFIG, 'r') as tc_file:
            self.config = json.load(tc_file)
        self.config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_processor = ImgProcessor(self.config["tensor_value_range"])
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])

        self.dataset_loaded = False
        self.models_dict = {}
        self.models_ready = False

    def load_dataset(self, dataset="MM", **dataset_kwargs):
        self.models_dict = {}
        self.models_ready = False
        dataset_class = DATASET_CLASSES[dataset]
        self.test_data = dataset_class.get_test(self.img_processor, **dataset_kwargs)
        self.config = update_cfg_from_dataset(self.config, self.test_data)
        print(f"INFO: loaded dataset '{self.test_data.NAME}' from {self.test_data.data_dir} "
              f"(action size: {self.test_data.ACTION_SIZE})")
        self.dataset_loaded = True

    def _prepare_testing(self, **testing_kwargs):
        raise NotImplementedError

    def load_models(self, model_dirs, ckpt_name="best_model.pth", cfg_name="run_cfg.json"):
        """
        overrides existing models
        desc: (model, model_cfg, preprocessing, postprocessing, test_metrics (initialized empty))
        """
        self.models_dict = {}
        for model_dir in model_dirs:
            model = torch.load(os.path.join(model_dir, ckpt_name)).to(self.config["device"])
            with open(os.path.join(model_dir, cfg_name), "r") as cfg_file:
                model_config = json.load(cfg_file)
            # get adapters to make model work with test cfg
            preprocessing, postprocessing = check_model_compatibility(self.config, model_config, model)
            self.models_dict[model.desc] = (model, model_config, preprocessing, postprocessing, [])
        self.models_ready = True

        # add baseline copy model
        clf_baseline = CopyLastFrame().to(self.config["device"])
        self.models_dict[clf_baseline.desc] = (clf_baseline, vars(self.config), nn.Identity(), nn.Identity(), [])

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
                    actions = data["actions"].to(self.config.device)

                    for (model, _, preprocess, postprocess, model_metrics_per_dp) in self.models_dict.values():
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
            for i, (model_desc, (model, model_cfg, _, _, _)) in enumerate(self.models_dict.items()):
                if model_desc == CopyLastFrame.desc:
                    continue  # don't print for copy baseline
                vis_out_dir = Path(model_cfg['out_dir']) / f"vis_{timestamp('test')}"
                vis_out_dir.mkdir()
                visualize_vid(self.test_data, self.config["context_frames"], self.config["pred_frames"], model, self.config["device"],
                              self.img_processor, vis_out_dir, vis_idx=vis_idx)

        # log or display metrics
        if eval_length > 0:
            wandb_full_suffix = "" if mini_test else "(full test)"
            models_dict_items = self.models_dict.items()
            for i, (model_desc, (_, model_cfg, _, _, model_metrics_per_dp)) in enumerate(models_dict_items):

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
                    wandb.init(config={"mini_test": mini_test, "model_dir": model_cfg['out_dir']},
                               project="vp-suite-testing", name=f"{model_desc}{wandb_full_suffix}", reinit=(i > 0))
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        wandb.log({"pred_frames": f+1, **mean_metric_dict})
                    if i == len(models_dict_items) - 1:
                        wandb.finish()

                # Per-model/per-pred.-horizon console log of the metrics
                else:
                    print("Printing out test results to terminal...")
                    print(f"\n{model_desc} (path: {model_cfg['out_dir']}): ")
                    for f, mean_metric_dict in enumerate(mean_metric_dicts):
                        print(f"pred_frames: {f + 1}")
                        for (k, v) in mean_metric_dict.items():
                            print(f" -> {k}: {v}")
