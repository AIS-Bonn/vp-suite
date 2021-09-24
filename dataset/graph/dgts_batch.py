import numpy as np
import torch
import torch.nn as nn

import torch_geometric
import torch_geometric_temporal
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


Edge_Indices = List[Union[np.ndarray, None]]
Edge_Weights = List[Union[np.ndarray, None]]
Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Batches = List[Union[np.ndarray, None]]



def from_data_list(cls, data_list: List[DynamicGraphTemporalSignal],
                   follow_batch: Optional[List[str]] = None,
                   exclude_keys: Optional[List[str]] = None):
    batch, slice_dict, inc_dict = collate(
        cls,
        data_list=data_list,
        increment=True,
        add_batch=True,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )

    batch._num_graphs = len(data_list)
    batch._slice_dict = slice_dict
    batch._inc_dict = inc_dict

    return batch

