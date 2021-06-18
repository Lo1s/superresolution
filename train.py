import argparse
import collections
import torch
from tqdm import tqdm

import loader.data_loaders as module_data
import model.srcnn.model as srcnn_model_arch
import model.srcnn.loss as srcnn_model_loss
import model.srcnn.metric as srcnn_model_metric
from parse_config import ConfigParser
from trainer.srcnn.trainer import Trainer
from utils import prepare_device


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', srcnn_model_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(srcnn_model_loss, config['loss'])
    metrics = [getattr(srcnn_model_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


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
#     # TODO: consider these plots
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



