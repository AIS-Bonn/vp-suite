from vp_suite.measure import METRIC_CLASSES

class PredictionMetricProvider():
    def __init__(self, config):

        self.device = config["device"]
        self.available_metrics = METRIC_CLASSES if config["metrics"] == "all" \
            else {k: METRIC_CLASSES[k] for k in config["metrics"]}
        if config["img_c"] not in [2, 3]:
            print("WARNING: 'FVD' measure won't be used since image channels needs to be in [2, 3]")
            self.available_metrics.pop("fvd")
        self.metrics = {k: metric(device=self.device) for k, metric in self.available_metrics.items()}

    def get_metrics(self, pred, target, frames=None, all_frame_cnts=False):
        '''
        input type: torch.tensor (torch.float)
        input shape: [b, t, c, h, w]
        If frames is specified, only considers the first 'frames' frames.
        '''

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
