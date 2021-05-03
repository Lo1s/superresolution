import os
from typing import Optional, Callable
import torch
from pathlib2 import Path


class SRNBA:

    project_path = Path.pwd()
    data_dir = 'data'
    data_path = project_path / data_dir
    data_file = 'data.pt'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(
            self,
            root: str,
            train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        self.root = root

        dataset = torch.load(os.path.join(self.processed_folder, self.data_file))
        self.data, self.labels = dataset['data'], dataset['labels']

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')