from torch import nn as nn


class BaseMeasure(nn.Module):
    r"""

    """
    NAME: str = NotImplemented  #: The clear-text name of the measure.
    REFERENCE: str = None  #: The reference where this measure is originally introduced.
    BIGGER_IS_BETTER = False  #: specifying whether bigger values are better
    OPT_VALUE = 0.  #: specifying the best value attainable (e.g. with equal tensors)

    def __init__(self, device):
        r"""

        Args:
            device ():
        """
        super(BaseMeasure, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, pred, target):
        r"""

        Args:
            pred ():
            target ():

        Returns:

        """
        pass

    @classmethod
    def to_display(cls, x):
        r"""

        Args:
            x ():

        Returns:

        """
        return x
