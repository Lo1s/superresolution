from torchvision import transforms
from base import BaseDataLoader
from datasets.srcnn.tutorial.srcnn_dataset import SRCNNDataset


class SRCNNDataloader(BaseDataLoader):
    """
    SRCNN data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = SRCNNDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)