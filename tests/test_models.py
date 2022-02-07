import pytest
import torch.cuda

from vp_suite.models import MODEL_CLASSES
from vp_suite.base.base_model import VideoPredictionModel

IMG_SHAPE = (3, 64, 64)
ACTION_SIZE = 3
TEMPORAL_DIM = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model input shapes
b = 2  # batch_size
c, h, w = IMG_SHAPE
p = 5  # pred_frames


@pytest.mark.parametrize('model_key', MODEL_CLASSES.keys(), ids=[v.NAME for v in MODEL_CLASSES.values()])
def test_models_without_actions(model_key):

    model_class = MODEL_CLASSES[model_key]
    model_kwargs = {
        "action_size": ACTION_SIZE,
        "img_shape": IMG_SHAPE,
        "temporal_dim": TEMPORAL_DIM,
        "action_conditional": False,
        "tensor_value_range": [0.0, 1.0]
    }
    model: VideoPredictionModel = model_class(DEVICE, **model_kwargs).to(DEVICE)
    t = p+3 if model_class.NEEDS_COMPLETE_INPUT else 3  # model.MIN_CONTEXT_FRAMES
    x = torch.randn(b, t, c, h, w, device=DEVICE)
    pred_1 = model.pred_1(x)
    pred_5, _ = model(x, pred_frames=p)
    assert pred_1.shape == (b, c, h, w)
    assert pred_5.shape == (b, 5, c, h, w)


@pytest.mark.parametrize('model_key', MODEL_CLASSES.keys(), ids=[v.NAME for v in MODEL_CLASSES.values()])
def test_models_with_actions(model_key):

    model_class = MODEL_CLASSES[model_key]
    model_kwargs = {
        "action_size": ACTION_SIZE,
        "img_shape": IMG_SHAPE,
        "temporal_dim": TEMPORAL_DIM,
        "action_conditional": model_class.CAN_HANDLE_ACTIONS,
        "tensor_value_range": [0.0, 1.0]
    }
    model: VideoPredictionModel = model_class(DEVICE, **model_kwargs).to(DEVICE)
    t_x = p+3 if model_class.NEEDS_COMPLETE_INPUT else 3  # model.MIN_CONTEXT_FRAMES
    t_a = p+3-1
    x = torch.randn(b, t_x, c, h, w, device=DEVICE)
    a = torch.randn(b, t_a, ACTION_SIZE, device=DEVICE)
    pred_1 = model.pred_1(x, actions=a)
    pred_5, _ = model(x, pred_frames=p, actions=a)

    print(x.shape, pred_1.shape, pred_5.shape)

    assert pred_1.shape == (b, c, h, w)
    assert pred_5.shape == (b, 5, c, h, w)
