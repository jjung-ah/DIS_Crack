# Coding by SUNN(01139138@hyundai-autoever.com)
# todo : to_tensor 추가
# todo : 어떻게 하면 parameter 들을 깔끔 + 통일 되게 가져올 수 있을까 고민

import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from . import TRANSFORMS_REGISTRY
from utils.types import Dictconfigs


@TRANSFORMS_REGISTRY.register()
# originally from https://github.com/xuebinqin/DIS
class GOSRandomHFlip(object):
    def __init__(self, parameters: Dictconfigs):
        self.prob = parameters.prob  # 0.5

    def __call__(self, img, gt):
        if random.random() >= self.prob:
            img = torch.flip(img, dims=[2])
            gt = torch.flip(gt, dims=[2])
        return img, gt


@TRANSFORMS_REGISTRY.register()
# originally from https://github.com/xuebinqin/DIS
class GOSResize(object):
    def __init__(self, parameters: Dictconfigs):
        self.size = (parameters.width, parameters.height)

    def __call__(self, img, gt):
        img = torch.squeeze(F.upsample(torch.unsqueeze(img, 0), self.size, mode='bilinear'), dim=0)
        gt = torch.squeeze(F.upsample(torch.unsqueeze(gt, 0), self.size, mode='bilinear'), dim=0)
        return img, gt


@TRANSFORMS_REGISTRY.register()
# originally from https://github.com/xuebinqin/DIS
class GOSRandomCrop(object):
    def __init__(self, parameters: Dictconfigs):
        self.size = (parameters.weight, parameters.height)

    def __call__(self, img, gt):
        h, w = img.shape[1:]
        new_w, new_h = self.size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[:, top:top+new_h, left:left+new_w]
        gt = gt[:, top:top+new_h, left:left+new_w]
        return img, gt


@TRANSFORMS_REGISTRY.register()
# originally from https://github.com/xuebinqin/DIS
class GOSNormalize(object):
    def __init__(self, parameters: Dictconfigs):
        self.mean = parameters.mean  # [0.485, 0.456, 0.406]
        self.std = parameters.std  # [0.229, 0.224, 0.225]

    def __call__(self, img, gt):
        img = normalize(img, self.mean, self.std)
        return img, gt
