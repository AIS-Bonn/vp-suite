import torch.nn as nn


class ModelBlock(nn.Module):
    NAME: str = __name__  #: The clear-text name for this model block.
    PAPER_REFERENCE = None  #: The publication where this model was introduced first.
    CODE_REFERENCE = None  #: The code location of the reference implementation.
    MATCHES_REFERENCE: str = None  #: A comment indicating whether the implementation in this package matches the reference.
