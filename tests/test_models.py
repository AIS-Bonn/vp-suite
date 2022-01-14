"""
def test_all_models(cfg):  # TODO re-write...
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
        model_class = MODEL_CLASSES.get(model_type, MODEL_CLASSES["copy"])
        model = model_class(config, **model_args).to(self.device)

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
"""