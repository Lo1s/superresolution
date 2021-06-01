import argparse
import collections
import time
from configparser import ConfigParser

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import h5py
import loader.data_loaders as module_data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets.srcnn.tutorial.srcnn_dataset import SRCNNDataset
from trainer.srcnn.trainer import train, validate
from model.srcnn.model import SRCNN

# plt.style.use('ggplot')
# # learning parameters TODO: move it to config file
# batch_size = 64
# epochs = 100
# lr = 0.001
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # inputs image dimensions
# img_rows, img_cols = 33, 33
# out_rows, out_cols = 33, 33
#
# data_dir = 'data/inputs/tutorial/train_mscale.h5'
# file = h5py.File(data_dir, mode='r')
# # in_train has shape (21824, 33, 33, 1) => 21824 images of size 33x33 with 1 color channel
# in_train = file['data'][:] # training data
# out_train = file['label'][:] # training labels
# file.close()
#
# # change the values to float32
# in_train = in_train.astype('float32')
# out_train = out_train.astype('float32')
# (x_train, x_val, y_train, y_val) = train_test_split(in_train, out_train, test_size=0.25)
# print('Training samples: ', x_train.shape[0])
# print('Validation samples: ', x_val.shape[0])
#
# # train and validation data
# train_data = SRCNNDataset(x_train, y_train)
# val_data = SRCNNDataset(x_val, y_val)
# # train and validation loaders
# train_loader = DataLoader(train_data, batch_size=batch_size)
# val_loader = DataLoader(val_data, batch_size=batch_size)
#
# # initialize the model
# print('Computation device: ', device)
# model = SRCNN().to(device)
# print(model)
# # optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr)
# # loss function
# criterion = nn.MSELoss()


# --- REFACTORED ---
def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='SRCNN')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in config.json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')

    ]
    config = ConfigParser.from_args(args, options)
    main(config)

# if __name__ == '__main__':
#     train_loss, val_loss = [], []
#     train_psnr, val_psnr = [], []
#     start = time.time()
#
#     # TODO: Use config for hyperparams
#     epochs = 1
#     for epoch in range(epochs):
#         print(f'Epoch {epoch + 1} of {epochs}')
#         train_epoch_loss, train_epoch_psnr = train(model, train_loader, train_data, optimizer, criterion, device)
#         val_epoch_loss, val_epoch_psnr = validate(model, val_loader, val_data, epoch, criterion, device)
#         print(f'Train PSNR: {train_epoch_psnr:.3f}')
#         print(f'Val PSNR: {val_epoch_psnr:.3f}')
#         train_loss.append(train_epoch_loss)
#         train_psnr.append(train_epoch_psnr)
#         val_loss.append(val_epoch_loss)
#         val_psnr.append(val_epoch_psnr)
#     end = time.time()
#     print(f'Finished training in: {((end - start) / 60):.3f} minutes')
#
#     # loss plots
#     plt.figure(figsize=(10, 7))
#     plt.plot(train_loss, color='orange', label='train loss')
#     plt.plot(val_loss, color='red', label='validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('data/outputs/loss.png')
#     plt.show()
#
#     # psnr plots
#     plt.figure(figsize=(10, 7))
#     plt.plot(train_psnr, color='green', label='train PSNR dB')
#     plt.plot(val_psnr, color='blue', label='validation PSNR dB')
#     plt.xlabel('Epochs')
#     plt.ylabel('PSNR (dB)')
#     plt.legend()
#     plt.savefig('data/outputs/psnr.png')
#     plt.show()
#
#     # save model to disk
#     print('Saving model...')
#     torch.save(model.state_dict(), 'data/saved/models/bw_tutorial_model.pth')



