from vp_suite.base.base_model import VideoPredictionModel


class CopyLastFrame(VideoPredictionModel):
    r"""

    """

    # model-specific constants
    NAME = "CopyLastFrame"
    REQUIRED_ARGS = []
    TRAINABLE = False

    def __init__(self, device=None, **model_kwargs):
        r"""

        Args:
            device ():
            **model_kwargs ():
        """
        super(CopyLastFrame, self).__init__(device, **model_kwargs)

    def pred_1(self, x, **kwargs):
        r"""

        Args:
            x ():
            **kwargs ():

        Returns:

        """
        return x[:, -1, :, :, :]
