import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from model.esrgan.utils.utils import MODEL_KEY, GENERATOR_KEY, DISCRIMINATOR_KEY
from test import save_predictions_as_imgs

# Load base low-resolution image.
fixed_lr = transforms.ToTensor()(Image.open(os.path.join("data/inputs/Set5", "butterfly.png"))).unsqueeze(0)


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, monitor_cfg_key='monitor',
                 epochs_cfg_key='epochs'):
        self.device = device
        self.fixed_lr = fixed_lr.to(self.device)

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.models = models
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizers = optimizers

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer[epochs_cfg_key]

        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get(monitor_cfg_key, 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            self.plot_epoch_result = cfg_trainer.get('plot_epoch_result', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        for i, model in enumerate(self.models):
            optimizer = self.optimizers[i]
            arch = type(model).__name__
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'monitor_best': self.mnt_best,
                'config': self.config
            }
            filename = str(self.checkpoint_dir / 'checkpoint-{}_epoch_{}.pth'.format(arch, epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint_path_parts = self.checkpoint_dir.parts

            if save_best:
                best_path = str(self.checkpoint_dir / f'model_{arch}_best.pth')
                torch.save(state, best_path)
                self.logger.info(f'Saving current best: model_{arch}_best.pth ...')

        # Each one epoch create a sr image.
        arch = type(self.models[MODEL_KEY]).__name__
        with torch.no_grad():
            sr = self.models[MODEL_KEY](self.fixed_lr)
            vutils.save_image(
                sr.detach(),
                os.path.join(self.checkpoint_dir, f'checkpoint-{arch}_epoch_{epoch}.png'),
                normalize=True
            )

    def _resume_checkpoint(self, resume_paths):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """

        for i, path in enumerate(resume_paths):
            self.logger.info("Loading checkpoint: {} ...".format(path))
            checkpoint = torch.load(path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

            if 'Generator' in checkpoint['arch']:
                key = GENERATOR_KEY
                arch_param = 'arch_esrgan_gen'
            elif 'Discriminator' in checkpoint['arch']:
                key = DISCRIMINATOR_KEY
                arch_param = 'arch_esrgan_disc'

            else:
                key = MODEL_KEY
                arch_param = 'arch_single'

            # load architecture params from checkpoint.
            if checkpoint['config'][arch_param] != self.config[arch_param]:
                self.logger.warning(
                    "Warning: Architecture configuration given in config file is different from that of "
                    "checkpoint. This may yield an exception while state_dict is being loaded.")
            self.models[key].load_state_dict(checkpoint['state_dict'])

            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning(
                    "Warning: Optimizer type given in config file is different from that of checkpoint. "
                    "Optimizer parameters not being resumed.")
            else:
                self.optimizers[key].load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
