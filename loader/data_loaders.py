from base import BaseDataLoader
from datasets.t91_patches.dataset import T91PatchesDataset


class T91PatchesDataloader(BaseDataLoader):
    """
    TP1 patches image data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = T91PatchesDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)