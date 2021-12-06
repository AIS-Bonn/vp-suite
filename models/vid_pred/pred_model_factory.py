import sys
sys.path.append(".")

import torch

from models.vid_pred.conv_lstm import LSTMModel, LSTMModelOld
from models.vid_pred.copy_last_frame import CopyLastFrameModel
from models.vid_pred.phydnet import PhyDNet
from models.vid_pred.st_lstm import STLSTMModel
from models.vid_pred.st_phy import STPhy
from models.vid_pred.unet_3d import UNet3dModel, UNet3dModelOld
from models.vid_pred.non_conv import NonConvLSTMModel
from models.vid_pred.lin_pred import SimpleV1, SimpleV2

MODELS = ["unet", "unet_old", "lstm", "lstm_old", "non_conv", "st_lstm", "copy", "phy", "st_phy", "simplev1", "simplev2"]

def get_pred_model(cfg):

    arch = cfg.pred_arch
    if cfg.include_actions:
        action_size = cfg.action_size
        print("using action-conditional video prediction if applicable")
    else:
        action_size = 0

    if arch == "simplev1":
        print("prediction model: SimpleV1")
        pred_model = SimpleV1(in_channels=cfg.num_channels, out_channels=cfg.num_channels, time_slice=5,
                              action_size=action_size, device=cfg.device)

    elif arch == "simplev2":
        print("prediction model: SimpleV2")
        pred_model = SimpleV2(in_channels=cfg.num_channels, out_channels=cfg.num_channels, time_slice=5,
                              action_size=action_size, device=cfg.device)

    elif arch == "unet":
        print("prediction model: UNet3d")
        pred_model = UNet3dModel(in_channels=cfg.num_channels, out_channels=cfg.num_channels, img_size=cfg.img_shape,
                                 time_dim=cfg.vid_input_length, features=cfg.pred_unet_features,
                                 action_size=action_size, device=cfg.device)

    elif arch == "unet_old":
        print("prediction model: UNet3d (old version)")
        pred_model = UNet3dModelOld(in_channels=cfg.num_channels, out_channels=cfg.num_channels,
                                    time_dim=cfg.vid_input_length, features=cfg.pred_unet_features)

    elif arch == "lstm":
        print("prediction model: LSTM")
        pred_model = LSTMModel(in_channels=cfg.num_channels, out_channels=cfg.num_channels, img_size=cfg.img_shape,
                               lstm_kernel_size=cfg.pred_lstm_kernel_size, num_layers=cfg.pred_lstm_num_layers,
                               action_size=action_size, device=cfg.device)

    elif arch == "lstm_old":
        print("prediction model: LSTM (old version)")
        pred_model = LSTMModelOld(in_channels=cfg.num_channels, out_channels=cfg.num_channels)

    elif arch == "non_conv":
        print("prediction model: Non-Conv LSTM")
        pred_model = NonConvLSTMModel(in_channels=cfg.num_channels, out_channels=cfg.num_channels,
                                      img_size=cfg.img_shape, lstm_kernel_size=(5, 5), num_layers=3,
                                      action_size=action_size, device=cfg.device)

    elif arch == "st_lstm":
        print("prediction model: ST-LSTM")
        pred_model = STLSTMModel(img_size=cfg.img_shape, img_channels=cfg.num_channels,
                                 enc_channels=cfg.pred_st_enc_channels,
                                 num_layers=cfg.pred_st_num_layers, action_size=action_size,
                                 inflated_action_dim=cfg.pred_st_inflated_action_dim,
                                 decouple_loss_scale=cfg.pred_st_decouple_loss_scale,
                                 reconstruction_loss_scale=cfg.pred_st_rec_loss_scale, device=cfg.device)

    elif arch == "phy":
        print("prediction model: PhyDNet")
        pred_model = PhyDNet(img_size=cfg.img_shape, img_channels=cfg.num_channels,
                             phy_cell_channels=cfg.pred_phy_enc_channels, phy_kernel_size=cfg.pred_phy_kernel_size,
                             moment_loss_scale=cfg.pred_phy_moment_loss_scale, action_size=action_size,
                             device=cfg.device)

    elif arch == "st_phy":
        print("prediction model: ST-Phy")
        pred_model = STPhy(img_size=cfg.img_shape, img_channels=cfg.num_channels,
                           enc_channels=cfg.pred_st_enc_channels, phy_channels=cfg.pred_phy_enc_channels,
                           num_layers=cfg.pred_st_num_layers, action_size=action_size,
                           inflated_action_dim=cfg.pred_st_inflated_action_dim,
                           phy_kernel_size=cfg.pred_phy_kernel_size,
                           decouple_loss_scale=cfg.pred_st_decouple_loss_scale,
                           reconstruction_loss_scale=cfg.pred_st_rec_loss_scale,
                           moment_loss_scale=cfg.pred_phy_moment_loss_scale, device=cfg.device)

    else:
        print("prediction model: CopyLastFrame")
        pred_model = CopyLastFrameModel()
        cfg.no_train = True

    total_params = sum(p.numel() for p in pred_model.parameters())
    trainable_params = sum(p.numel() for p in pred_model.parameters() if p.requires_grad)
    print(f"Model parameters (total / trainable): {total_params} / {trainable_params}")
    return pred_model.to(cfg.device)


def test_all_models(cfg):
    import time
    from itertools import product

    cfg.num_channels = cfg.dataset_classes+1 if cfg.include_actions else cfg.dataset_classes
    cfg.img_shape = 135, 240
    cfg.action_size = 3

    x = torch.randn((cfg.batch_size, cfg.vid_input_length, cfg.num_channels, *cfg.img_shape)).to(cfg.device)
    a = torch.randn((cfg.batch_size, cfg.vid_total_length, cfg.action_size)).to(cfg.device)

    for (include_actions, arch) in product([False, True], MODELS):
        cfg.include_actions = include_actions
        cfg.pred_arch = arch
        model = get_pred_model(cfg)

        print("")
        print(f"Checking {model.__class__.__name__} (action-conditional: {getattr(model, 'use_actions', False)})")
        print(f"Parameter count (total / learnable): {sum([p.numel() for p in model.parameters()])}"
              f" / {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        t_start = time.time()
        pred1, _ = model(x, actions=a)
        t_pred1 = round(time.time() - t_start, 6)

        t_start = time.time()
        preds, _ = model.pred_n(x, cfg.vid_pred_length, actions=a)
        t_preds = round(time.time() - t_start, 6)

        print(f"Pred time (1 out frame / {cfg.vid_pred_length} out frames): {t_pred1}s / {t_preds}s")
        print(f"Shapes ({cfg.vid_input_length} in frames / 1 out frame / {cfg.vid_pred_length} out frames): "
              f"{list(x.shape)} / {list(pred1.shape)} / {list(preds.shape)}")