import pytest
from vp_suite import VPSuite
from vp_suite.dataset import DATASET_CLASSES

@pytest.mark.slow
@pytest.mark.parametrize('dataset', DATASET_CLASSES.keys(), ids=[v.NAME for v in DATASET_CLASSES.values()])
def test_dataset(dataset):
    suite = VPSuite()
    suite.load_dataset(dataset, "train")
    train_wrapper = suite.datasets[-2]
    train_wrapper.set_seq_len(1, 1, 1)
    suite.load_dataset(dataset, "test")
    test_wrapper = suite.datasets[-1]
    test_wrapper.set_seq_len(1, 1, 1)
    assert train_wrapper.frame_shape == test_wrapper.frame_shape
    assert train_wrapper.action_size == test_wrapper.action_size
    assert set(train_wrapper.datasets.keys()) == {"main", "train", "val"}, f"{dataset}"
    assert set(test_wrapper.datasets.keys()) == {"main", "test"}, f"{dataset}"
    example_data = [train_wrapper.train_data[0], train_wrapper.val_data[0],
                    test_wrapper.test_data[0]]  # train, val, test
    for ex_ in example_data:
        assert isinstance(ex_, dict), f"{dataset}"
        assert set(ex_.keys()) == {"frames", "actions"}, f"{dataset}"
        assert ex_["frames"].shape[1] == train_wrapper.frame_shape[-1], f"{dataset}"
        assert ex_["frames"].shape[2:] == train_wrapper.frame_shape[:2], f"{dataset}"
        assert ex_["actions"].shape[-1] == train_wrapper.action_size, f"{dataset}"
    assert len(suite.training_sets) == 1
    assert len(suite.test_sets) == 1
