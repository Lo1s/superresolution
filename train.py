import argparse
import collections
import torch
from torchvision import models
from tqdm import tqdm

import loader.data_loaders as module_data
import model.srcnn.model as srcnn_model_arch
import model.srcnn.loss as srcnn_model_loss
import model.srcnn.metric as srcnn_model_metric
import model.unet.model as unet_model_arch
import model.unet.loss as unet_model_loss
from model.unet.loss import create_loss_model
from parse_config import ConfigParser
from trainer.srcnn.trainer import Trainer
from utils import prepare_device


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    #model = config.init_obj('arch', srcnn_model_arch)
    model = config.init_obj('arch', unet_model_arch)
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
                      lr_scheduler=lr_scheduler,
                      logging=False,
                      use_vgg_loss=True)
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



