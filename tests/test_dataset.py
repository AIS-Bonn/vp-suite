import pytest
from vp_suite.dataset import DATASET_CLASSES
from vp_suite.dataset._wrapper import DatasetWrapper
from vp_suite.utils.img_processor import ImgProcessor

@pytest.mark.slow
@pytest.mark.parametrize('dataset_class', DATASET_CLASSES.keys(), ids=[v.NAME for v in DATASET_CLASSES.values()])
def test_dataset(dataset_class):

    img_processor = ImgProcessor(value_min=0.0, value_max=1.0)
    train_wrapper = DatasetWrapper(dataset_class, img_processor, "train")
    train_wrapper.set_seq_len(1, 1, 1)
    test_wrapper = DatasetWrapper(dataset_class, img_processor, "test")
    test_wrapper.set_seq_len(1, 1, 1)
    assert train_wrapper.frame_shape == test_wrapper.frame_shape
    assert train_wrapper.action_size == test_wrapper.action_size
    assert set(train_wrapper.datasets.keys()) == {"main", "train", "val"}
    assert set(test_wrapper.datasets.keys()) == {"main", "test"}
    example_data = [train_wrapper.train_data[0], train_wrapper.val_data[0],
                    test_wrapper.test_data[0]]  # train, val, test
    for ex_ in example_data:
        assert isinstance(ex_, dict)
        assert set(ex_.keys()) == {"frames", "actions"}
        assert ex_["frames"].shape[1] == train_wrapper.frame_shape[-1]
        assert ex_["frames"].shape[2:] == train_wrapper.frame_shape[:2]
        assert ex_["actions"].shape[-1] == train_wrapper.action_size
