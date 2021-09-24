from typing import Union, List

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, HeteroData, Dataset, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal as SGTS
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS
from torch_geometric_temporal.signal import DynamicGraphStaticSignal as DGSS
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch as DGTSBatch
from dataset.graph.dgts_batch import from_data_list as DGTSBatch_from_data_list


def collate_DGTS(
        data_list: List[DGTS],
        increment: bool = True,
        add_batch: bool = True,
        follow_batch: Optional[Union[List[str]]] = None,
        exclude_keys: Optional[Union[List[str]]] = None,
) -> Tuple[BaseData, Mapping, Mapping]:

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # TODO indices, increments?

    all_graphs = [[g for g in iter(signal)] for signal in data_list]  # list (samples) of list(temporal) of torch_geometric Data
    if len(set([len(sample_list) for sample_list in all_graphs])) > 1:  # check for consistent sequence lengths
        raise ValueError("Batch Samples of differing sequence length are currently not supported.")
    batched_edge_indices = []
    batched_edge_weights = []
    batched_node_features = []
    batched_targets = []

    for t in range(len(all_graphs[0])):  # collate features of graphs of the same timestep
        batched_edge_indices.append(torch.cat([signal[t]["edge_index"] for signal in all_graphs], dim=1))
        batched_edge_weights.append(torch.cat([signal[t]["edge_attr"] for signal in all_graphs], dim=0))
        batched_node_features.append(torch.cat([signal[t]["x"] for signal in all_graphs], dim=0))
        batched_targets.append(torch.cat([signal[t]["y"] for signal in all_graphs], dim=0))

    return DynamicGraphTemporalSignal(
        edge_indices=batched_edge_indices,
        edge_weights=batched_edge_weights,
        features=batched_node_features,
        targets=batched_targets
    )

# TODO de-batch?


def DGTSBatch_from_list(data_list: Union[List[Data], List[HeteroData]],
                        follow_batch: Optional[List[str]] = None,
                        exclude_keys: Optional[List[str]] = None):
    batch, slice_dict, inc_dict = collate_DGTS(
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


class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, SGTS):
            raise NotImplementedError  # TODO
        elif isinstance(elem, DGTS):
            return DGTSBatch_from_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, DGSS):
            raise NotImplementedError  # TODO
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


[docs]
class DataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: object,  # TODO
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str] = [],
        exclude_keys: List[str] = [],
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=Collater(follow_batch, exclude_keys), **kwargs)