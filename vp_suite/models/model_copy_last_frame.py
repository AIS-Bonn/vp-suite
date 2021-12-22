from vp_suite.models.base_model import VideoPredictionModel


class CopyLastFrame(VideoPredictionModel):

    trainable = False

    @classmethod
    def model_desc(cls):
        return "CopyLastFrame"

    def __init__(self, cfg=None):
        super(CopyLastFrame, self).__init__(cfg)

    def forward(self, x, **kwargs):
        return x[:, -1, :, :, :], None
