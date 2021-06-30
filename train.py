import argparse
import collections
import math
import torch.nn as nn
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
import model.esrgan.discriminator as esrgan_model_arch
from model.esrgan.loss import ContentLoss
from parse_config import ConfigParser
from trainer.esrgan.gan_trainer import ESRGANTrainer
from trainer.esrgan.psnr_trainer import PSNRTrainer
from trainer.srcnn.trainer import SRCNNTrainer
from utils import prepare_device, configure


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    device, device_ids = prepare_device(config['n_gpu'])
    if config['is_gan']:
        generator = configure(arch="esrgan16", pretrained=True, logger=logger)
        discriminator = config.init_obj('arch_esrgan_disc', esrgan_model_arch)
        logger.info(generator)
        logger.info(discriminator)
        generator.to(device)
        discriminator.to(device)
        # prepare for (multi-device) GPU training
        if len(device_ids) > 1:
            generator = torch.nn.DataParallel(generator, device_ids=device_ids)
            discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

        # All optimizer function and scheduler function.
        psnr_epochs = config['trainer']['psnr_epochs']
        psnr_lr = config['trainer']['psnr_lr']
        gan_epochs = config['trainer']['gan_epochs']
        gan_lr = config['trainer']['gan_lr']

        # Loss = 0.01 * pixel loss + content loss + 0.005 * adversarial loss
        pixel_criterion = nn.L1Loss().to(device)
        content_criterion = ContentLoss().to(device)
        adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

        psnr_optimizer = torch.optim.Adam(generator.parameters(), lr=psnr_lr, betas=(0.9, 0.999))
        psnr_epoch_indices = math.floor(psnr_epochs // 4)
        psnr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(psnr_optimizer, psnr_epoch_indices, 1, 1e-7)
        interval_epoch = math.ceil(gan_epochs // 8)
        epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
        # Discriminator
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan_lr, betas=(0.9, 0.999))
        discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, epoch_indices, 0.5)
        # Generator
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=gan_lr, betas=(0.9, 0.999))
        generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, epoch_indices, 0.5)

        train_metrics = ['psnr']
        valid_metrics = ['psnr', 'ssim', 'gmsd']
        psnr_trainer = PSNRTrainer(
            generator,
            pixel_criterion,
            train_metrics,
            valid_metrics,
            psnr_optimizer,
            config,
            device,
            data_loader,
            valid_data_loader,
            lr_scheduler=psnr_scheduler,
            logging=False,
            monitor_cfg_key='monitor_psnr'
        )
        psnr_trainer.train()

        train_metrics = ['pixel_loss', 'content_loss', 'adversarial_loss', 'ssim']
        valid_metrics = ['pixel_loss', 'content_loss', 'psnr', 'ssim', 'gmsd']
        esrgan_trainer = ESRGANTrainer(
            generator,
            discriminator,
            generator_optimizer, discriminator_optimizer,
            pixel_criterion, content_criterion, adversarial_criterion,
            train_metrics,
            valid_metrics,
            config,
            device,
            data_loader,
            valid_data_loader,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            logging=False,
            monitor_cfg_key='monitor'
        )
        esrgan_trainer.train()

    else:
        # build model architecture, then print to console
        # model = config.init_obj('arch_srcnn', srcnn_model_arch)
        model = config.init_obj('arch', unet_model_arch)
        logger.info(model)
        # prepare for (multi-device) GPU training
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
        trainer = SRCNNTrainer(model, criterion, metrics, optimizer,
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
