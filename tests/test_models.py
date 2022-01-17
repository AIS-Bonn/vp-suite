import pytest
import torch.cuda

from vp_suite.models import MODEL_CLASSES
from vp_suite.models._base_model import VideoPredictionModel

IMG_SHAPE = (64, 64, 3)
ACTION_SIZE = 3
TEMPORAL_DIM = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

@pytest.mark.slow
@pytest.mark.parametrize('model_key', MODEL_CLASSES.keys(), ids=[v.NAME for v in MODEL_CLASSES.values()])
def test_models_without_actions(model_key):

    model_class = MODEL_CLASSES[model_key]
    model_args = {
        "action_size": ACTION_SIZE,
        "img_shape": IMG_SHAPE,
        "temporal_dim": TEMPORAL_DIM,
        "action_conditional": False
    }
    model : VideoPredictionModel = model_class(DEVICE, **model_args).to(DEVICE)
    b = 2
    t = model.min_context_frames
    h, w, c = IMG_SHAPE
    x = torch.randn(b, t, c, h, w)
    pred_1 = model.pred_1(x)
    pred_5 = model(x, pred_length = 5)

    assert pred_1.shape == (b, 1, c, h, w)
    assert pred_5.shape == (b, 5, c, h, w)


@pytest.mark.slow
@pytest.mark.parametrize('model_key', MODEL_CLASSES.keys(), ids=[v.NAME for v in MODEL_CLASSES.values()])
def test_models_with_actions(model_key):

    model_class = MODEL_CLASSES[model_key]
    model_args = {
        "action_size": ACTION_SIZE,
        "img_shape": IMG_SHAPE,
        "temporal_dim": TEMPORAL_DIM,
        "action_conditional": model_class.CAN_HANDLE_ACTIONS
    }
    model : VideoPredictionModel = model_class(DEVICE, **model_args).to(DEVICE)
    b = 2
    t = model.min_context_frames
    h, w, c = IMG_SHAPE
    x = torch.randn(b, t, c, h, w)
    a = torch.randn(b, t, ACTION_SIZE)
    pred_1 = model.pred_1(x, actions=a)
    pred_5 = model(x, pred_length = 5, actions=a)

    assert pred_1.shape == (b, 1, c, h, w)
    assert pred_5.shape == (b, 5, c, h, w)
