from torch.utils.data import Subset



class DatasetWrapper:
    """ A class that wraps torch Datasets to handle training and testing data in the same way """

    ALLOWED_SPLITS = ["train", "test"]

    def __init__(self, dataset_class, img_processor, split, **dataset_kwargs):
        super(DatasetWrapper, self).__init__()

        assert split in self.ALLOWED_SPLITS, "TODO"
        if split == "train":
            train_data, val_data = dataset_class.get_train_val(img_processor, **dataset_kwargs)
            main_data = train_data.dataset if isinstance(train_data, Subset) else train_data
            self.datasets = {
                "main": main_data,
                "train": train_data,
                "val": val_data
            }
        else:
            test_data = dataset_class.get_test(img_processor, **dataset_kwargs)
            self.datasets = {
                "main": test_data,
                "test": test_data
            }
        self.is_ready = False  # set to true after seq_len has been set (pre-requisite for training)

    def is_training_set(self):
        return "train" in self.datasets.keys() and "val" in self.datasets.keys()

    def is_test_set(self):
        return "test" in self.datasets.keys()

    @property
    def train_data(self):
        return self.datasets.get("train", None)

    @property
    def val_data(self):
        return self.datasets.get("val", None)

    @property
    def test_data(self):
        return self.datasets.get("test", None)

    @property
    def NAME(self):
        return self.datasets["main"].NAME

    @property
    def data_dir(self):
        return self.datasets["main"].data_dir

    @property
    def action_size(self):
        return self.datasets["main"].action_size

    @property
    def frame_shape(self):
        return self.datasets["main"].frame_shape

    @property
    def config(self):
        return self.datasets["main"].config

    @property
    def img_processor(self):
        return self.datasets["main"].img_processor

    def set_seq_len(self, context_frames, pred_frames, seq_step):
        self.datasets["main"].set_seq_len(context_frames, pred_frames, seq_step)
        # set the seq_len for val_data aswell if it's a separate dataset
        if self.is_training_set() and not getattr(self.val_data, "ready_for_usage", True):
            self.val_data.set_seq_len(context_frames, pred_frames, seq_step)
        self.is_ready = True