from vp_suite.models._base_model import VideoPredictionModel


class CopyLastFrame(VideoPredictionModel):

    NAME = "CopyLastFrame"
    TRAINABLE = False

    def __init__(self, dataset_config=None, device=None, **model_args):
        super(CopyLastFrame, self).__init__(dataset_config, device, **model_args)

    def pred_1(self, x, **kwargs):
        return x[:, -1, :, :, :], None
