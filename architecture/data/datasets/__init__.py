# Coding by SUNN(01139138@hyundai-autoever.com)

from fvcore.common.registry import Registry

DATASETS_REGISTRY = Registry("DATASETS")
DATASETS_REGISTRY.__doc__ = "Registry for Data-Sets."


# datasets.
from .press import PressDataset

__all__ = [
    'PressDataset'
]
