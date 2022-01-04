import sys, os, random, json
sys.path.append("")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn

from vp_suite.models.copy_last_frame import CopyLastFrame
from vp_suite.measure.metric_provider import PredictionMetricProvider
from vp_suite.utils.utils import timestamp, check_model_compatibility
from vp_suite.utils.visualization import visualize_vid
from vp_suite.utils.img_processor import ImgProcessor
from vp_suite.dataset._factory import create_test_dataset, update_cfg_from_dataset


def test(test_cfg):

    # prep
    random.seed(test_cfg.seed)
    np.random.seed(test_cfg.seed)
    torch.manual_seed(test_cfg.seed)
    test_cfg.img_processor = ImgProcessor(test_cfg.tensor_value_range)

    # DATASET
    test_data, test_loader = create_test_dataset(test_cfg)
    test_cfg = update_cfg_from_dataset(test_cfg, test_data)

    # MODELS
    models_dict = {}  # desc: (model, model_cfg, preprocessing, postprocessing, test_metrics (initialized empty))
    for vis_out_dir in test_cfg.model_dirs:
        model = torch.load(os.path.join(vis_out_dir, "best_model.pth")).to(test_cfg.device)
        with open(os.path.join(vis_out_dir, "run_cfg.json"), "r") as cfg_file:
            model_cfg = json.load(cfg_file)
        # get adapters to make model work with test cfg
        if test_cfg.context_frames is None or test_cfg.pred_frames is None:
            print("INFO: context frames and/or pred_frames unspecified -> will default to models' respective values")
        preprocessing, postprocessing = check_model_compatibility(test_cfg, model_cfg, model)
        models_dict[model.desc] = (model, model_cfg, preprocessing, postprocessing, [])

    # add baseline copy model
    clf_baseline = CopyLastFrame().to(test_cfg.device)
    models_dict[clf_baseline.desc] = (clf_baseline, vars(test_cfg), nn.Identity(), nn.Identity(), [])

    iter_loader = iter(test_loader)
    eval_length = 10 if test_cfg.mini_test else len(iter_loader)

    # evaluation / metric calc.
    if eval_length > 0:
        with torch.no_grad():

            metric_provider = PredictionMetricProvider(test_cfg)

            for i in tqdm(range(eval_length)):
                data = next(iter_loader)
                img_data = data["frames"].to(test_cfg.device)
                input = img_data[:, :test_cfg.context_frames]
                target = img_data[:, test_cfg.context_frames:test_cfg.total_frames]
                actions = data["actions"].to(test_cfg.device)

                for (model, _, preprocess, postprocess, model_metrics_per_dp) in models_dict.values():
                    input = preprocess(input)  # test format to model format
                    if getattr(model, "use_actions", False):
                        pred, _ = model.pred_n(input, pred_length=test_cfg.pred_frames, actions=actions)
                    else:
                        pred, _ = model.pred_n(input, pred_length=test_cfg.pred_frames)
                    pred = postprocess(pred)  # model format to test format
                    cur_metrics = metric_provider.get_metrics(pred, target, all_frame_cnts=True)
                    model_metrics_per_dp.append(cur_metrics)

    # save visualizations
    if not test_cfg.no_vis:
        print(f"Saving visualizations for trained models...")
        num_vis = 5
        vis_idx = np.random.choice(len(test_data), num_vis, replace=False)
        for i, (model_desc, (model, model_cfg, _, _, _)) in enumerate(models_dict.items()):
            if model_desc == CopyLastFrame.desc: continue  # don't print for copy baseline
            vis_out_dir = Path(model_cfg['out_dir']) / f"vis_{timestamp('test')}"
            vis_out_dir.mkdir()
            visualize_vid(test_data, test_cfg.context_frames, test_cfg.pred_frames, model, test_cfg.device,
                          test_cfg.img_processor, vis_out_dir, vis_idx=vis_idx)

    # log or display metrics
    if eval_length > 0:
        wandb_full_suffix = "" if test_cfg.mini_test else "(full test)"
        models_dict_items = models_dict.items()
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
            if not test_cfg.no_wandb:
                print("Logging test results to WandB for all models...")
                wandb.init(config={"mini_test": test_cfg.mini_test, "model_dir": model_cfg['out_dir']},
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
