# Coding by SUNN(01139138@hyundai-autoever.com)

from typing import List, Dict

from utils.types import Dictconfigs
from .functions import TRANSFORMS_REGISTRY


def build_transforms(configs: Dictconfigs):
    transform_config = configs.transforms
    funcs = []
    for name, parameters in transform_config.items():
        func = TRANSFORMS_REGISTRY.get(name)(parameters)
        funcs.append(func)
    return Compose(funcs)


class Compose(object):
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        '''
        :param sample: dict(idx=idx, image=image, gt=gt, shape=shape)
        :return: dict(idx=idx, image=image, gt=gt, shape=shape)
        '''
        for t in self.transforms:
            sample = t(sample)
        return sample
