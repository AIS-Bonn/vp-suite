import sys
sys.path.append(".")

import torch

from models.prediction.conv_lstm import LSTMModel
from models.prediction.copy_last_frame import CopyLastFrameModel
from models.prediction.phydnet.phydnet import PhyDNet
from models.prediction.st_lstm.st_lstm import STLSTMModel
from models.prediction.st_lstm.st_phy import STPhy
from models.prediction.unet_3d import UNet3dModel

from run import DEVICE


def get_pred_model(cfg, num_channels, video_in_length, device):

    if cfg.include_actions:
        action_size = cfg.action_size
        print("using action-conditional video prediction if applicable")
    else:
        action_size = 0

    if cfg.pred_arch == "unet":
        print("prediction model: UNet3d")
        pred_model = UNet3dModel(in_channels=num_channels, out_channels=num_channels, time_dim=video_in_length,
                                 features=cfg.pred_unet_features)

    elif cfg.pred_arch == "lstm":
        print("prediction model: LSTM")
        pred_model = LSTMModel(in_channels=num_channels, out_channels=num_channels)

    elif cfg.pred_arch == "st_lstm":
        print("prediction model: ST-LSTM")
        pred_model = STLSTMModel(img_size=cfg.img_shape, img_channels=num_channels,
                                 enc_channels=cfg.pred_st_enc_channels,
                                 num_layers=cfg.pred_st_num_layers, action_size=action_size,
                                 inflated_action_dim=cfg.pred_st_inflated_action_dim,
                                 decouple_loss_scale=cfg.pred_st_decouple_loss_scale,
                                 reconstruction_loss_scale=cfg.pred_st_reconstruction_loss_scale, device=device)

    elif cfg.pred_arch == "phy":
        print("prediction model: PhyDNet")
        pred_model = PhyDNet(img_size=cfg.img_shape, img_channels=num_channels, action_size=action_size, device=device)

    elif cfg.pred_arch == "st_phy":
        print("prediction model: ST-Phy")
        pred_model = STPhy(img_size=cfg.img_shape, img_channels=num_channels, action_size=action_size, device=device)

    else:
        print("prediction model: CopyLastFrame")
        pred_model = CopyLastFrameModel()
        cfg.no_train = True

    total_params = sum(p.numel() for p in pred_model.parameters())
    trainable_params = sum(p.numel() for p in pred_model.parameters() if p.requires_grad)
    print(f"Model parameters (total / trainable): {total_params} / {trainable_params}")
    return pred_model.to(device)


def test_all_models():

    import time

    batch_size = 6
    time_dim = 5
    num_channels = 23
    pred_length = 5
    img_size = 135, 240
    action_size = 3
    # TODO
    raise NotImplementedError
    x = torch.randn((batch_size, time_dim, num_channels, *img_size)).to(DEVICE)
    a = torch.randn((batch_size, time_dim+pred_length, action_size)).to(DEVICE)

    models = [
        CopyLastFrameModel(),
        UNet3dModel(in_channels=num_channels, out_channels=num_channels, time_dim=time_dim).to(DEVICE),
        LSTMModel(in_channels=num_channels, out_channels=num_channels).to(DEVICE),
        STLSTMModel(img_size, img_channels=num_channels, device=DEVICE, action_size=0),
        STLSTMModel(img_size, img_channels=num_channels, device=DEVICE, action_size=action_size),
        PhyDNet(img_size, img_channels=num_channels, device=DEVICE, action_size=0),
        PhyDNet(img_size, img_channels=num_channels, device=DEVICE, action_size=action_size),
        STPhy(img_size, img_channels=num_channels, device=DEVICE, action_size=0),
        STPhy(img_size, img_channels=num_channels, device=DEVICE, action_size=action_size)
    ]

    for model in models:
        print("")
        print(f"Checking {model.__class__.__name__} (action-conditional: {getattr(model, 'use_actions', False)})")
        print(f"Parameter count (total / learnable): {sum([p.numel() for p in model.parameters()])}"
              f" / {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        t_start = time.time()
        pred1, _ = model(x, actions=a)
        t_pred1 = round(time.time() - t_start, 6)

        t_start = time.time()
        preds, _ = model.pred_n(x, pred_length, actions=a)
        t_preds = round(time.time() - t_start, 6)

        print(f"Pred time (1 out frame / {pred_length} out frames): {t_pred1}s / {t_preds}s")
        print(f"Shapes ({time_dim} in frames / 1 out frame / {pred_length} out frames): "
              f"{list(x.shape)} / {list(pred1.shape)} / {list(preds.shape)}")


if __name__ == '__main__':
    test_all_models()