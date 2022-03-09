import warnings
import torch
from vp_suite.measure import LOSS_CLASSES


class PredictionLossProvider:
    r"""
        This class provides bundled access to multiple losses. With this class's :meth:`get_losses()` method,
        all specified loss values are calculated on the same input prediction and target tensor.

        Attributes:
            device (str): A string specifying whether to use the GPU for calculations (`cuda`) or the CPU (`cpu`).
            losses (dict): The concrete instantiated losses that the loss provider uses when provided with input tensors.
    """
    def __init__(self, config: dict):
        r"""
        Initializes the provider by extracting device and loss IDs from the provided config dict
        and instantiating the losses that shall be used.

        Args:
            config (dict): A dictionary containing the devices and losses to use. The provided losses come with the scales that should be multiplied by the respective loss value.
        """
        self.device = config["device"]
        loss_scales = config["losses_and_scales"]
        if "fvd" in loss_scales.keys() and config["img_c"] not in [2, 3]:
            warnings.warn("'FVD' measure won't be used since image channels needs to be in [2, 3]")
            loss_scales.pop("fvd")
        self.losses = {k: (LOSS_CLASSES[k](device=self.device), scale) for k, scale in loss_scales.items()}

    def get_losses(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Takes in tensors of predicted frames and the corresponding ground truth and calculates the losses for
        the loss classes instantiated previously. Each loss

        Args:
            pred (torch.Tensor): The predicted frame sequence as a 5D float tensor (batch, frames, c, h, w).
            target (torch.Tensor): The ground truth frame sequence as a 5D float tensor (batch, frames, c, h, w)

        Returns:
            1. A dictionary containing the loss IDs and the corresponding values of each loss in display representation.

            2. The sum of scaled losses (total loss).
        """
        if pred.shape != target.shape:
            raise ValueError("Output images and target images are of different shape!")

        loss_display_values, total_loss = {}, torch.tensor(0.0, device=self.device)
        for key, (loss, scale) in self.losses.items():
            val = loss(pred, target)
            total_loss += scale * val
            loss_display_values[key] = loss.to_display(val)

        return loss_display_values, total_loss
