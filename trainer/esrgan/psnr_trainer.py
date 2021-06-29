import numpy as np
import torch
import os
import torch.nn as nn
import torch.cuda.amp as amp
import torchvision.utils as vutils
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

from torchvision import models
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from base import BaseTrainer
from model.esrgan.utils import MODEL_KEY, GENERATOR_KEY
from model.esrgan.utils.calculate_gmsd import GMSD
from model.esrgan.utils.calculate_ssim import SSIM
from model.srcnn.metric import psnr
from model.unet.loss import create_loss_model
from utils import inf_loop, MetricTracker


class PSNRTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, train_metric_ftns, valid_metric_ftns, optimizer, config, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, logging=True,
                 monitor_cfg_key='monitor_psnr', epochs_cfg_key='psnr_epochs'):
        super().__init__([model], criterion, train_metric_ftns, [optimizer], config, device,
                         monitor_cfg_key=monitor_cfg_key, epochs_cfg_key=epochs_cfg_key)

        self.scaler = amp.GradScaler()
        
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.logging = logging

        self.train_metric_ftns = train_metric_ftns
        self.train_metrics = MetricTracker('loss', *[m for m in self.train_metric_ftns], writer=self.writer)
        self.valid_metric_ftns = valid_metric_ftns
        self.valid_metrics = MetricTracker('loss', *[m for m in self.valid_metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        mse_loss = nn.MSELoss().to(self.device).eval()
        self.models[GENERATOR_KEY].train()
        self.train_metrics.reset()
        for batch_idx, (lr, hr) in enumerate(tqdm(self.data_loader)):
            lr, hr = lr.to(self.device), hr.to(self.device)

            # Start mixed precision training.
            self.optimizers[GENERATOR_KEY].zero_grad()

            with amp.autocast():
                sr = self.models[GENERATOR_KEY](lr)
                loss = self.criterion(sr, hr)

            psnr = 10 * torch.log10(1. / mse_loss(sr, hr))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizers[GENERATOR_KEY])
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('psnr', psnr)

            if batch_idx % self.log_step == 0 and self.logging:
                self.logger.debug('\nTrain Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(lr.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        mse_loss = nn.MSELoss().to(self.device).eval()
        # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
        ssim_loss = SSIM().to(self.device).eval()
        # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
        gmsd_loss = GMSD().to(self.device).eval()

        self.models[GENERATOR_KEY].eval()
        self.valid_metrics.reset()
        progress_bar = tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader))
        total_psnr_value = 0.
        total_ssim_value = 0.
        total_gmsd_value = 0.

        with torch.no_grad():
            for i, (lr, hr) in progress_bar:
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.models[GENERATOR_KEY](lr)
                loss = self.criterion(sr, hr)

                # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
                total_psnr_value += 10 * torch.log10(1. / mse_loss(sr, hr))
                # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
                total_ssim_value += ssim_loss(sr, hr)
                # The GMSD of the generated fake high-resolution image and real high-resolution image is calculated.
                total_gmsd_value += gmsd_loss(sr, hr)

                progress_bar.set_description(f"PSNR: {total_psnr_value / (i + 1):.2f} "
                                             f"SSIM: {total_ssim_value / (i + 1):.4f} "
                                             f"GMSD: {total_gmsd_value / (i + 1):.4f}")

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + i, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('psnr', total_psnr_value / (i + 1))
                self.valid_metrics.update('ssim', total_ssim_value / (i + 1))
                self.valid_metrics.update('gmsd', total_gmsd_value / (i + 1))

                self.writer.add_image('input', make_grid(lr.cpu(), nrow=8, normalize=True))

        # add histogram of the model parameters to the tensorboard
        for name, p in self.models[GENERATOR_KEY].named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
