# Coding by SUNN(01139138@hyundai-autoever.com)

import os
import glob
import yaml
from typing import List, Dict


def load_yaml(config_path: str) -> Dict:
    with open(config_path, encoding='utf-8') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs


def get_all_items(directory: str, img_ext: List) -> List:
    if not os.path.exists(directory):
        raise FileNotFoundError(f'There is no folder for {directory}.')

    file_list = []
    for ext in img_ext:
        file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
    return sorted(list(set(file_list)))
