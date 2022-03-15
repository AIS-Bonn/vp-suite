import torch.nn as nn


class VPModelBlock(nn.Module):
    r"""
    This class implements the base class for many model blocks used by this package's prediction models.
    The named constants exposed in this base class provide convenient ways to implement model blocks that match
    official implementations.
    """
    NAME: str = __name__  #: The clear-text name for this model block.
    PAPER_REFERENCE = None  #: The publication where this model was introduced first.
    CODE_REFERENCE = None  #: The code location of the reference implementation.
    MATCHES_REFERENCE: str = None  #: A comment indicating whether the implementation in this package matches the reference.
