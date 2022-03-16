from torch.utils.data import Subset


class VPDatasetWrapper:
    r"""
    A class that wraps VPDataset instances to handle training and testing data in the same way.
    """

    ALLOWED_SPLITS = ["train", "test"]  #: On creation, the wrapper must be initialized with one of these split values.

    def __init__(self, dataset_class, split, **dataset_kwargs):
        r"""
        Instantiates the dataset class specified by the given dataset_class value and split identifier.

        Args:
            dataset_class (Any): A string identifier corresponding to the dataset class that should be instantiated.
            split (str): A string specifying whether this Wrapper should wrap a training/validation or a test set.
            **dataset_kwargs (Any): Additional optional dataset configuration arguments.
        """
        super(VPDatasetWrapper, self).__init__()

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        r"""
        Returns a string representation of the Wrapper and the wrapped dataset(s).
        """
        return f"DatasetWrapper[{self.NAME}](datasets={self.datasets}, is_ready={self.is_ready})"

    def is_training_set(self):
        r"""
        Returns True if this wrapper wraps training/validation data, false otherwise.
        """
        return "train" in self.datasets.keys() and "val" in self.datasets.keys()

    def is_test_set(self):
        r"""
        Returns True if this wrapper wraps testing fata, false otherwise.
        """
        return "test" in self.datasets.keys()

    @property
    def train_data(self):
        r"""
        Returns the wrapped training dataset. If not existent, raises an error.
        """
        train_data = self.datasets.get("train", None)
        if train_data is None:
            raise KeyError(f"dataset '{self.NAME}' does not contain training data")
        return train_data

    @property
    def val_data(self):
        r"""
        Returns the wrapped validation dataset. If not existent, raises an error.
        """
        val_data = self.datasets.get("val", None)
        if val_data is None:
            raise KeyError(f"dataset '{self.NAME}' does not contain validation data")
        return val_data

    @property
    def test_data(self):
        r"""
        Returns the wrapped test dataset. If not existent, raises an error.
        """
        test_data = self.datasets.get("test", None)
        if test_data is None:
            raise KeyError(f"dataset '{self.NAME}' does not contain test data")
        return test_data

    @property
    def NAME(self):
        r"""
        Returns the dataset name.
        """
        return self.datasets["main"].NAME

    @property
    def data_dir(self):
        r"""
        Returns the dataset's data location
        """
        return self.datasets["main"].data_dir

    @property
    def action_size(self):
        r"""
        Returns the dataset's action size (which is 0 for datasets that don't provide actions)
        """
        return self.datasets["main"].ACTION_SIZE

    @property
    def img_shape(self):
        r"""
        Returns the shape of each frame of the sequences that will be provided by the dataset (after preprocessing)
        """
        return self.datasets["main"].img_shape

    @property
    def config(self):
        r"""
        Returns a dictionary containing the dataset's configuration parameters.
        """
        return self.datasets["main"].config

    def set_seq_len(self, context_frames, pred_frames, seq_step):
        r"""
        Sets the desired sequence length for all wrapped datasets by calculating the needed sequence length from the
        provided sequence parameters

        Args:
            context_frames (int): Number of context/input frames (these will be provided as input to the VPModel)
            pred_frames (int): Number of frames the VPModel has to predict, given the context frames.
            seq_step (int): Sequence step (assembling the sequence from every Nth frame or the original video)
        """
        self.datasets["main"].set_seq_len(context_frames, pred_frames, seq_step)

        # set the seq_len for val_data as well if it's a separate dataset
        if self.is_training_set() and not isinstance(self.val_data, Subset):
            self.val_data.set_seq_len(context_frames, pred_frames, seq_step)
        self.is_ready = True

    def reset_rng(self):
        r"""
        Resets the RNG for all wrapped datasets.
        """
        self.datasets["main"].reset_rng()
        if self.is_training_set() and not isinstance(self.val_data, Subset):
            self.val_data.reset_rng()
