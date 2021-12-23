from vp_suite.measure.image_wise import MSE, L1, SmoothL1, LPIPS, SSIM, PSNR
from vp_suite.measure.fvd.fvd import FrechetVideoDistance as FVD

METRICS = {
    "mse": MSE,
    "l1": L1,
    "smooth_l1": SmoothL1,
    "lpips": LPIPS,
    "ssim": SSIM,
    "psnr": PSNR,
    "fvd": FVD
}

AVAILABLE_METRICS = METRICS.keys()

class PredictionMetricProvider():
    def __init__(self, cfg):

        self.device = cfg.device
        self.available_metrics = METRICS if cfg.metrics == "all" else {k: METRICS[k] for k in cfg.metrics}
        if cfg.img_c not in [2, 3]:
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
        frames = frames or pred.shape[1]

        metrics = []
        frames = [frames] if not all_frame_cnts else range(1, frames + 1)
        for frame_cnt in frames:
            pred_ = pred[:, :frame_cnt]
            target_ = target[:, :frame_cnt]
            frame_cnt_metrics = {
                f"{key} ({'↑' if metric.bigger_is_better else '↓'})":
                    metric.to_display(metric(pred_, target_).item())
                for key, metric in self.metrics.items()
            }
            # remove metrics that returned 'None' (e.g. because they don't support the current frame cnt
            frame_cnt_metrics = {k: v for k, v in frame_cnt_metrics.items() if v is not None}
            metrics.append(frame_cnt_metrics)

        return metrics
