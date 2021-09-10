from models.prediction.conv_lstm import LSTMModel
from models.prediction.copy_last_frame import CopyLastFrameModel
from models.prediction.pred_model import test
from models.prediction.st_lstm import STLSTMModel
from models.prediction.unet_3d import UNet3dModel

from config import DEVICE


def get_pred_model(cfg, num_channels, video_in_length, device):

    if cfg.model == "unet":
        print("prediction model: UNet3d")
        pred_model = UNet3dModel(in_channels=num_channels, out_channels=num_channels, time_dim=video_in_length).to(device)

    elif cfg.model == "lstm":
        print("prediction model: LSTM")
        pred_model = LSTMModel(in_channels=num_channels, out_channels=num_channels).to(device)

    elif cfg.model == "st_lstm":
        if cfg.include_actions:
            a_dim = train_data.action_size
            print("prediction model: action-conditional ST-LSTM")
        else:
            a_dim = 0
            print("prediction model: ST-LSTM")
        pred_model = STLSTMModel(img_size=train_data.img_shape, img_channels=num_channels, action_size=a_dim, device=device)

    else:
        print("prediction model: CopyLastFrame")
        pred_model = CopyLastFrameModel().to(DEVICE)
        cfg.no_train = True

    return pred_model