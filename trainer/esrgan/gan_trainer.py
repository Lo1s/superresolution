import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from pathlib import Path

from torchvision import models
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from base import BaseTrainer
from model.esrgan.utils import SSIM, GMSD, GENERATOR_KEY, DISCRIMINATOR_KEY
from model.srcnn.metric import psnr
from model.unet.loss import create_loss_model
from utils import inf_loop, MetricTracker


class ESRGANTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer,
                 pixel_criterion, content_criterion, adversarial_criterion, train_metric_ftns, valid_metric_ftns,
                 config, device, data_loader, valid_data_loader=None, generator_scheduler=None,
                 discriminator_scheduler=None, len_epoch=None, logging=True, monitor_cfg_key='monitor',
                 epochs_cfg_key='gan_epochs'):
        super().__init__([generator, discriminator], [pixel_criterion, content_criterion, adversarial_criterion],
                         train_metric_ftns, [generator_optimizer, discriminator_optimizer], config, device,
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
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.logging = logging

        self.pixel_criterion = pixel_criterion
        self.content_criterion = content_criterion
        self.adversarial_criterion = adversarial_criterion

        self.train_metric_ftns = train_metric_ftns
        self.train_metrics = MetricTracker('loss', *[m for m in self.train_metric_ftns], writer=self.writer)
        self.valid_metric_ftns = valid_metric_ftns
        self.valid_metrics = MetricTracker(*[m for m in self.valid_metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        ssim_loss = SSIM().to(self.device).eval()
        self.models[GENERATOR_KEY].train()  # generator
        self.models[DISCRIMINATOR_KEY].train()  # discriminator

        self.train_metrics.reset()
        for batch_idx, (lr, hr) in enumerate(tqdm(self.data_loader)):
            lr, hr = lr.to(self.device), hr.to(self.device)

            batch_size = lr.size(0)

            # The real sample label is 1, and the generated sample label is 0.
            real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype)
            fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype)
            real_label, fake_label = real_label.to(self.device), fake_label.to(self.device)

            ##############################################
            # (1) Update D network: E(hr)[fake(C(D(hr) - E(sr)C(sr)))] + E(sr)[fake(C(fake) - E(real)C(real))]
            ##############################################
            # Set discriminator gradients to zero.
            self.optimizers[DISCRIMINATOR_KEY].zero_grad()

            with amp.autocast():
                sr = self.models[GENERATOR_KEY](lr)
                # It makes the discriminator distinguish between real sample and fake sample.
                real_output = self.models[DISCRIMINATOR_KEY](hr)
                fake_output = self.models[DISCRIMINATOR_KEY](sr.detach())

                # Adversarial loss for real and fake images (relativistic average GAN)
                d_loss_real = self.adversarial_criterion(real_output - torch.mean(fake_output), real_label)
                d_loss_fake = self.adversarial_criterion(fake_output - torch.mean(real_output), fake_label)

                # Count all discriminator losses.
                d_loss = d_loss_real + d_loss_fake

            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizers[DISCRIMINATOR_KEY])
            self.scaler.update()

            ##############################################
            # (2) Update G network: E(hr)[sr(C(D(hr) - E(sr)C(sr)))] + E(sr)[sr(C(fake) - E(real)C(real))]
            ##############################################
            self.optimizers[GENERATOR_KEY].zero_grad()

            with amp.autocast():
                sr = self.models[GENERATOR_KEY](lr)
                # It makes the discriminator unable to distinguish the real samples and fake samples.
                real_output = self.models[DISCRIMINATOR_KEY](hr.detach())
                fake_output = self.models[DISCRIMINATOR_KEY](sr)

                # Calculate the absolute value of pixels with L1 loss.
                pixel_loss = self.pixel_criterion(sr, hr.detach())
                # The 35th layer in VGG19 is used as the feature extractor by default.
                content_loss = self.content_criterion(sr, hr.detach())
                # Adversarial loss for real and fake images (relativistic average GAN)
                adversarial_loss = self.adversarial_criterion(fake_output - torch.mean(real_output), real_label)

                # Count all generator losses.
                g_loss = 0.01 * pixel_loss + 1 * content_loss + 0.005 * adversarial_loss
                ssim = ssim_loss(sr, hr)

            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.optimizers[GENERATOR_KEY])
            self.scaler.update()

            # Set generator gradients to zero.
            self.optimizers[GENERATOR_KEY].zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', g_loss.item())
            self.train_metrics.update('ssim', ssim.item())
            self.train_metrics.update('pixel_loss', pixel_loss.item())
            self.train_metrics.update('content_loss', content_loss.item())
            self.train_metrics.update('adversarial_loss', adversarial_loss.item())

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.generator_scheduler is not None:
            self.generator_scheduler.step()
        if self.discriminator_scheduler is not None:
            self.discriminator_scheduler.step()
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

                pixel_loss = self.pixel_criterion(sr, hr.detach())
                content_loss = self.content_criterion(sr, hr.detach())

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
                self.valid_metrics.update('pixel_loss', pixel_loss.item())
                self.valid_metrics.update('content_loss', content_loss.item())
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
