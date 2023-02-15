# Coding by SUNN(01139138@hyundai-autoever.com)

import torch
import numpy as np
from typing import Dict
from skimage import io

from . import DATASETS_REGISTRY
from utils.types import Tensor


def read_img(img_dir: str) -> Tensor:
    img = io.imread(img_dir)

    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # np.array to torch.tensor
    img = torch.tensor(img, dtype=torch.float32)
    img = torch.transpose(torch.transpose(img, 1, 2), 0, 1)
    return img


def read_gt(gt_dir: str) -> Tensor:
    gt = io.imread(gt_dir)

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    # np.array to torch.tensor
    torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
    return gt


@DATASETS_REGISTRY.register()
class PressDataset(object):
    '''
    Basic Dataset-Class of this repo.
    '''
    def __init__(self, datasets: Dict, transform=None):
        self.datasets = datasets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> Dict:
        '''
        return Dict : set(str), img_dir(str), gt_dir(str), img(tensor), gt(tensor)
        '''
        data = self.datasets[idx]  # set, img_dir, gt_dir

        img = read_img(img_dir=data['img_dir'])
        gt = read_gt(gt_dir=data['gt_dir'])

        # apply transform.
        img = torch.divide(img, 255.0)
        gt = torch.divide(gt, 255.0)
        if self.transform:
            img, gt = self.transform(img, gt)
        data.update(dict(img=img, gt=gt))
        return data
