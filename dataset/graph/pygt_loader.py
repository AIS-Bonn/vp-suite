from typing import Union, List, Optional

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, HeteroData, Dataset, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal as SGTS
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS
from torch_geometric_temporal.signal import DynamicGraphStaticSignal as DGSS
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch as SGTSBatch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch as DGTSBatch
from torch_geometric_temporal.signal import DynamicGraphStaticSignalBatch as DGSSBatch


def collate_temporal_signal(
        signal_list: List[Union[SGTS, DGTS, DGSS]],
        signal_batch_class: Union[SGTSBatch, DGTSBatch, DGSSBatch],
        follow_batch: Optional[Union[List[str]]] = None,
        exclude_keys: Optional[Union[List[str]]] = None,
):

    # check for inconsistent input signal types
    if len(set([type(signal) for signal in signal_list])) > 1:
        raise ValueError("Given Signal objects are of differing type!")

    # determine output type
    in_type = type(signal_list[0])
    signal_batch_type = None
    if in_type == SGTS:
        signal_batch_type = SGTSBatch
    elif in_type == DGTS:
        signal_batch_type = DGTSBatch
    elif in_type == DGSS:
        signal_batch_type = DGSSBatch

    # create list(samples) of list(temporal sequence) of torch_geometric.Data/HeteroData
    all_graphs = [[g for g in iter(signal)] for signal in signal_list]

    # check for inconsistent sequence lengths
    if len(set([len(sample_list) for sample_list in all_graphs])) > 1:
        raise ValueError("Batch Samples of differing sequence length are currently not supported.")

    # create diagonalized torch_geometric.Batch objects by timestep
    batches_by_timestep = [Batch.from_data_list([sample[t] for sample in all_graphs], follow_batch, exclude_keys)
                    for t in range(len(all_graphs[0]))]

    # assemble signal of batched graphs
    return signal_batch_type(
        edge_indices = [batch["edge_index"].numpy() for batch in batches_by_timestep],
        edge_weights = [batch["edge_attr"].numpy() for batch in batches_by_timestep],
        features = [batch["x"].numpy() for batch in batches_by_timestep],
        targets = [batch["y"].numpy() for batch in batches_by_timestep],
        batches = [batch["batch"].numpy() for batch in batches_by_timestep]
    )

class PYGTCollater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, SGTS) or isinstance(elem, DGTS) or isinstance(elem, DGSS):
            return collate_temporal_signal(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, Data) or isinstance(elem, HeteroData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
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

class DataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData], List[SGTS], List[DGTS], List[DGSS]],
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
                         collate_fn=PYGTCollater(follow_batch, exclude_keys), **kwargs)