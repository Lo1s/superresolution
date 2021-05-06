# TODO: refactor train script from trainer into root .py file
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import h5py
from torchvision.utils import save_image
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets.srcnn.tutorial.srcnn_dataset import SRCNNDataset
from model.srcnn.metric import psnr
from model.srcnn.model import SRCNN

plt.style.use('ggplot')
# learning parameters TODO: move it to config file
batch_size = 64
epochs = 100
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# inputs image dimensions
img_rows, img_cols = 33, 33
out_rows, out_cols = 33, 33

file = h5py.File('../../data/inputs/tutorial/train_mscale.h5', mode='r')
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
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.MSELoss()


def train(model, dataloader):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)

        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)

        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()

        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every 'batch_size' iterations)
        batch_psnr = psnr(label, outputs)
        running_psnr += batch_psnr

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(train_data) / dataloader.batch_size)
    return final_loss, final_psnr


def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label)

            # add loss to each item (total items in a batch = batch size)
            running_loss += loss.item()
            # calculate batch psnr (once every 'batch_size' iterations)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
        outputs = outputs.cpu()
        save_image(outputs, f'../../data/outputs/val_sr{epoch}.png')

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(val_data) / dataloader.batch_size)
    return final_loss, final_psnr