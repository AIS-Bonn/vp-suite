import pytest
from vp_suite.datasets import DATASET_CLASSES
from vp_suite.utils.dataset_wrapper import VPDatasetWrapper

@pytest.mark.parametrize('dataset_str', DATASET_CLASSES.keys(), ids=[v.NAME for v in DATASET_CLASSES.values()])
def test_dataset(dataset_str):

    dataset_class = DATASET_CLASSES[dataset_str]
    train_wrapper = VPDatasetWrapper(dataset_class, "train")
    train_wrapper.set_seq_len(1, 1, 1)
    test_wrapper = VPDatasetWrapper(dataset_class, "test")
    test_wrapper.set_seq_len(1, 1, 1)
    assert train_wrapper.img_shape == test_wrapper.img_shape
    assert train_wrapper.action_size == test_wrapper.action_size
    assert set(train_wrapper.datasets.keys()) == {"main", "train", "val"}
    assert set(test_wrapper.datasets.keys()) == {"main", "test"}
    example_data = [train_wrapper.train_data[0], train_wrapper.val_data[0],
                    test_wrapper.test_data[0]]  # train, val, test
    for ex_ in example_data:
        assert isinstance(ex_, dict)
        assert set(ex_.keys()) == {"frames", "actions", "origin"}
        assert ex_["frames"].shape[-3:] == train_wrapper.img_shape
        if train_wrapper.action_size > 0:
            assert ex_["actions"].shape[-1] == train_wrapper.action_size
