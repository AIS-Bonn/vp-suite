from vp_suite.models.base_model import VideoPredictionModel


class CopyLastFrameModel(VideoPredictionModel):

    def __init__(self):
        super(CopyLastFrameModel, self).__init__()

    @classmethod
    def model_desc(cls):
        return "Copy_last_frame"

    def forward(self, x, **kwargs):
        return x[:, -1, :, :, :], None
