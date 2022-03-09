import warnings

import torch

from vp_suite.measure import METRIC_CLASSES


class PredictionMetricProvider:
    r"""
        This class provides bundled access to multiple metrics. With this class's :meth:`get_metrics()` method,
        all specified metric scores are calculated on the same input prediction and target tensor.

        Attributes:
            device (str): A string specifying whether to use the GPU for calculations (`cuda`) or the CPU (`cpu`).
            available_metrics (dict): A dictionary containing the string identifiers and corresponding metrics that the metric provider should use when provided with input tensors.
            metrics (dict): The concrete instantiated metrics that the metric provider uses when provided with input tensors.
    """
    def __init__(self, config: dict):
        r"""
        Initializes the provider by extracting device and metric IDs from the provided config dict
        and instantiating the metrics that shall be used.

        Args:
            config (dict): A dictionary containing the devices and metrics to use.
        """
        self.device = config["device"]
        self.available_metrics = METRIC_CLASSES if config["metrics"] == "all" \
            else {k: METRIC_CLASSES[k] for k in config["metrics"]}
        if config["img_c"] not in [2, 3]:
            warnings.warn("'FVD' measure won't be used since image channels needs to be in [2, 3]")
            self.available_metrics.pop("fvd")
        self.metrics = {k: metric(device=self.device) for k, metric in self.available_metrics.items()}

    def get_metrics(self, pred: torch.Tensor, target: torch.Tensor, frames: int = None, all_frame_cnts: bool = False):
        r"""
        Takes in tensors of predicted frames and the corresponding ground truth and calculates the metric scores for
        the metrics instantiated previously.

        Args:
            pred (torch.Tensor): The predicted frame sequence as a 5D float tensor (batch, frames, c, h, w).
            target (torch.Tensor): The ground truth frame sequence as a 5D float tensor (batch, frames, c, h, w)

            frames (int): If frames is specified, only considers the first 'frames' frames.
            all_frame_cnts (bool): If set to true, elicits metrics for all prediction horizons from 1 up to the maximum number of frames. Otherwise, just elicits metrics for the specified number of frames

        Returns:
            A list of dictionaries, where each dictionary contains the metric ids
            and the corresponding result value for a specific number of prediction frames.
        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError("Input tensors expected to be 5-dimensional!")
        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")
        pred = pred.contiguous()
        target = target.contiguous()
        frames = frames or pred.shape[1]

        metrics = []
        frames = [frames] if not all_frame_cnts else range(1, frames + 1)
        for frame_cnt in frames:
            pred_ = pred[:, :frame_cnt]
            target_ = target[:, :frame_cnt]
            frame_cnt_metrics = dict()
            for key, metric in self.metrics.items():
                metric_val = metric(pred_, target_)
                if metric_val is not None:
                    frame_cnt_metrics[f"{key} ({'↑' if metric.BIGGER_IS_BETTER else '↓'})"] \
                        = metric.to_display(metric_val.item())
            # remove metrics that returned 'None' (e.g. because they don't support the current frame cnt
            frame_cnt_metrics = {k: v for k, v in frame_cnt_metrics.items() if v is not None}
            metrics.append(frame_cnt_metrics)

        return metrics
