# Coding by SUNN(01139138@hyundai-autoever.com)

from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = "Registry for Model."


# models.
from .isnet import ISNetDIS, ISNetGTEncoder

__all__ = [
    'ISNetDIS',
    'ISNetGTEncoder'
]
