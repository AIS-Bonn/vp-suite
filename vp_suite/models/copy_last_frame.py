from vp_suite.models.base_model import VideoPredictionModel


class CopyLastFrameModel(VideoPredictionModel):

    trainable = False

    @classmethod
    def model_desc(cls):
        return "CopyLastFrame"

    def __init__(self, cfg):
        super(CopyLastFrameModel, self).__init__(cfg)

    def forward(self, x, **kwargs):
        return x[:, -1, :, :, :], None
