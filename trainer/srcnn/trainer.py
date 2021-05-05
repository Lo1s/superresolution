# TODO: refactor train script from trainer into root .py file
import torch
import matplotlib
import matplotlib.pyplot as plt
import h5py

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets.srcnn.tutorial.srcnn_dataset import SRCNNDataset
from model.srcnn.model import SRCNN

plt.style.use('ggplot')
# learning parameters TODO: move it to config file
batch_size = 64
epochs = 100
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# input image dimensions
img_rows, img_cols = 33, 33
out_rows, out_cols = 33, 33

file = h5py.File('../../data/input/tutorial/train_mscale.h5', mode='r')
# in_train has shape (21824, 33, 33, 1) => 21824 images of size 33x33 with 1 color channel
in_train = file['data'][:] # training data
out_train = file['label'][:] # training labels
file.close()

# change the values to float32
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')
(x_train, x_val, y_train, y_val) = train_test_split(in_train, out_train, test_size=0.25)
print('Training samples: ', x_train.shape[0])
print('Validation samples: ', x_val.shape[0])

# train and validation data
train_data = SRCNNDataset(x_train, y_train)
val_data = SRCNNDataset(x_val, y_val)
# train and validation loaders
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

# initialize the model
print('Computation device: ', device)
model = SRCNN().to(device)
print(model)
