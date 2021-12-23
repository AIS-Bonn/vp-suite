import sys
sys.path.append("")

import torch

from vp_suite.models.model_convlstm import ConvLSTM
from vp_suite.models.model_copy_last_frame import CopyLastFrame
from vp_suite.models.model_phydnet import PhyDNet
from vp_suite.models.model_st_lstm import STLSTM
from vp_suite.models.model_st_phy import STPhy
from vp_suite.models.model_unet3d import UNet3D
from vp_suite.models.model_lstm import LSTM
from vp_suite.models.model_simple import SimpleV1, SimpleV2

pred_models = {
    "unet": UNet3D,
    "lstm" : ConvLSTM,
    "non_conv" : LSTM,
    "st_lstm" : STLSTM,
    "copy" : CopyLastFrame,
    "phy" : PhyDNet,
    "st_phy" : STPhy,
    "simplev1": SimpleV1,
    "simplev2": SimpleV2,
}

AVAILABLE_MODELS = pred_models.keys()

def create_pred_model(cfg):
    model_class = pred_models.get(cfg.model_type, pred_models["copy"])
    ac_str = "(action-conditional)" if cfg.use_actions and model_class.can_handle_actions else ""
    print(f"Creating prediction model '{model_class.model_desc()}' {ac_str}")
    pred_model = model_class(cfg).to(cfg.device)
    if not pred_model.trainable:
        cfg.no_train = True

    total_params = sum(p.numel() for p in pred_model.parameters())
    trainable_params = sum(p.numel() for p in pred_model.parameters() if p.requires_grad)
    print(f"Model parameters (total / trainable): {total_params} / {trainable_params}")
    return pred_model.to(cfg.device)

def test_all_models(cfg):  # TODO
    import time
    from itertools import product

    cfg.img_shape = 3, 135, 240
    cfg.img_c, cfg.img_h, cfg.img_w = cfg.img_shape
    cfg.action_size = 3

    x = torch.randn((cfg.batch_size, cfg.context_frames, cfg.img_c, cfg.img_h, cfg.img_w)).to(cfg.device)
    a = torch.randn((cfg.batch_size, cfg.total_frames, cfg.action_size)).to(cfg.device)

    for (use_actions, arch) in product([False, True], AVAILABLE_MODELS):
        cfg.use_actions = use_actions
        cfg.pred_arch = arch
        model = create_pred_model(cfg)

        print("")
        print(f"Checking {model.__class__.__name__} (action-conditional: {getattr(model, 'use_actions', False)})")
        print(f"Parameter count (total / learnable): {sum([p.numel() for p in model.parameters()])}"
              f" / {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        t_start = time.time()
        pred1, _ = model(x, actions=a)
        t_pred1 = round(time.time() - t_start, 6)

        t_start = time.time()
        preds, _ = model.pred_n(x, cfg.pred_frames, actions=a)
        t_preds = round(time.time() - t_start, 6)

        print(f"Pred time (1 out frame / {cfg.pred_frames} out frames): {t_pred1}s / {t_preds}s")
        print(f"Shapes ({cfg.context_frames} in frames / 1 out frame / {cfg.pred_frames} out frames): "
              f"{list(x.shape)} / {list(pred1.shape)} / {list(preds.shape)}")