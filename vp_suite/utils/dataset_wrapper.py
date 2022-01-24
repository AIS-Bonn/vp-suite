from torch.utils.data import Subset

class DatasetWrapper:
    r"""A class that wraps torch Datasets to handle training and testing data in the same way

    """

    ALLOWED_SPLITS = ["train", "test"]  #: TODO

    def __init__(self, dataset_class, split, **dataset_kwargs):
        r"""

        Args:
            dataset_class ():
            split ():
            **dataset_kwargs ():
        """
        super(DatasetWrapper, self).__init__()

        if split == "train":
            train_data, val_data = dataset_class.get_train_val(**dataset_kwargs)
            main_data = train_data.dataset if isinstance(train_data, Subset) else train_data
            self.datasets = {
                "main": main_data,
                "train": train_data,
                "val": val_data
            }
        elif split == "test":
            test_data = dataset_class.get_test(**dataset_kwargs)
            self.datasets = {
                "main": test_data,
                "test": test_data
            }
        else:
            raise ValueError(f"parameter {split} needs to be one of the following: {self.ALLOWED_SPLITS}")
        self.is_ready = False  # set to true after seq_len has been set (pre-requisite for training)

    def is_training_set(self):
        r"""

        Returns:

        """
        return "train" in self.datasets.keys() and "val" in self.datasets.keys()

    def is_test_set(self):
        r"""

        Returns:

        """
        return "test" in self.datasets.keys()

    @property
    def train_data(self):
        r"""

        Returns:

        """
        return self.datasets.get("train", None)

    @property
    def val_data(self):
        r"""

        Returns:

        """
        return self.datasets.get("val", None)

    @property
    def test_data(self):
        r"""

        Returns:

        """
        return self.datasets.get("test", None)

    @property
    def NAME(self):
        r"""

        Returns:

        """
        return self.datasets["main"].NAME

    @property
    def data_dir(self):
        r"""

        Returns:

        """
        return self.datasets["main"].data_dir

    @property
    def action_size(self):
        r"""

        Returns:

        """
        return self.datasets["main"].ACTION_SIZE

    @property
    def frame_shape(self):
        r"""

        Returns:

        """
        return self.datasets["main"].DATASET_FRAME_SHAPE

    @property
    def config(self):
        r"""

        Returns:

        """
        return self.datasets["main"].config

    def set_seq_len(self, context_frames, pred_frames, seq_step):
        r"""

        Args:
            context_frames ():
            pred_frames ():
            seq_step ():

        Returns:

        """
        self.datasets["main"].set_seq_len(context_frames, pred_frames, seq_step)

        # set the seq_len for val_data as well if it's a separate dataset
        if self.is_training_set() and not getattr(self.val_data, "ready_for_usage", True):
            self.val_data.set_seq_len(context_frames, pred_frames, seq_step)
        self.is_ready = True
