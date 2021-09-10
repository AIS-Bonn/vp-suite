from models.prediction.pred_model import VideoPredictionModel


class CopyLastFrameModel(VideoPredictionModel):

    def __init__(self):
        super(CopyLastFrameModel, self).__init__()

    def forward(self, x, **kwargs):
        return x[:, -1, :, :, :], None