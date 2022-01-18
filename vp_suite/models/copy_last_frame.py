from vp_suite.models._base_model import VideoPredictionModel


class CopyLastFrame(VideoPredictionModel):

    # model-specific constants
    NAME = "CopyLastFrame"
    REQUIRED_ARGS = []
    TRAINABLE = False

    def __init__(self, device=None, **model_args):
        super(CopyLastFrame, self).__init__(device, **model_args)

    def pred_1(self, x, **kwargs):
        return x[:, -1, :, :, :]
