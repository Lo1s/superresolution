import torch
from torch.utils.data import Dataset


class SRCNNDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )