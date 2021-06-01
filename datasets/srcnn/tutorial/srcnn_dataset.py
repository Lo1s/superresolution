import torch
import h5py
from torch.utils.data import Dataset


class SRCNNDataset(Dataset):
    def __init__(self, root):
        file = h5py.File(root, mode='r')
        # in_train has shape (21824, 33, 33, 1) => 21824 images of size 33x33 with 1 color channel
        in_train = file['data'][:]  # training data
        out_train = file['label'][:]  # training labels
        file.close()

        # change the values to float32
        self.data = in_train.astype('float32')
        self.targets = out_train.astype('float32')

        # TODO: add download, train, transform options

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )