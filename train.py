import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets.srcnn.tutorial.srcnn_dataset import SRCNNDataset
from trainer.srcnn.trainer import train, validate
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

file = h5py.File('data/inputs/tutorial/train_mscale.h5', mode='r')
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

if __name__ == '__main__':
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()

    # TODO: Use config for hyperparams
    epochs = 100
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} of {epochs}')
        train_epoch_loss, train_epoch_psnr = train(model, train_loader, train_data, optimizer, criterion, device)
        val_epoch_loss, val_epoch_psnr = validate(model, val_loader, val_data, epoch, criterion, device)
        print(f'Train PSNR: {train_epoch_psnr:.3f}')
        print(f'Val PSNR: {val_epoch_psnr:.3f}')
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
    end = time.time()
    print(f'Finished training in: {((end - start) / 60):.3f} minutes')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('data/outputs/loss.png')
    plt.show()

    # psnr plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('data/outputs/psnr.png')
    plt.show()

    # save model to disk
    print('Saving model...')
    torch.save(model.state_dict(), 'data/outputs/model.pth')



