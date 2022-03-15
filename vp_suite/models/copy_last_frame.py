from vp_suite.base import VPModel


class CopyLastFrame(VPModel):
    r"""
    A simple, non-trainable baseline model
    that simply returns the latest frame as the next predicted frame.
    """
    NAME = "CopyLastFrame"
    REQUIRED_ARGS = []
    TRAINABLE = False

    def __init__(self, device=None, **model_kwargs):
        super(CopyLastFrame, self).__init__(device, **model_kwargs)

    def pred_1(self, x, **kwargs):
        return x[:, -1, :, :, :]
