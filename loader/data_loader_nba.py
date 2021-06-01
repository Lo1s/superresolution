from torchvision import transforms
from base import BaseDataLoader
from datasets import SRNBA


class SRDataLoader(BaseDataLoader):
    """
    Super Resolution data loading
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([
            transforms.ToTensor() # TODO: normalize dataset
        ])
        self.data_dir = data_dir
        self.dataset = SRNBA(self.data_dir, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)