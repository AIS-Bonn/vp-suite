import sys
sys.path.append("")

from vp_suite.evaluation.image_loss import LPIPS, MSE, L1, SSIM, PSNR


class PredictionMetricProvider():
    def __init__(self, cfg):

        self.device = cfg.device
        self.metrics = {
            "mse": MSE(device=self.device),
            "mae": L1(device=self.device),
            #"smooth_l1": SmoothL1(device=self.device),
            "lpips": LPIPS(device=self.device),
            "ssim": SSIM(device=self.device),
            "psnr": PSNR(device=self.device)
        }

    def get_metrics(self, pred, target, frames=None, all_frame_cnts=False):
        '''
        input type: torch.tensor (torch.float)
        input shape: [b, t, c, h, w]
        input range: [-1.0, 1.0]
        if frames is specified, only considers the first 'frames' frames.
        '''

        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")

        frames = frames or pred.shape[1]
        pred = pred[:, :frames]
        target = target[:, :frames]

        metrics = []
        frames = [frames] if not all_frame_cnts else range(1, frames + 1)

        for frame_cnt in frames:
            pred_ = pred[:, :frame_cnt]
            target_ = target[:, :frame_cnt]
            frame_cnt_metrics = {
                f"{key} ({'↑' if loss.bigger_is_better else '↓'})":
                    loss.loss_to_display(loss(pred_, target_).item())
                for key, loss in self.metrics.items()
            }
            metrics.append(frame_cnt_metrics)

        return metrics
