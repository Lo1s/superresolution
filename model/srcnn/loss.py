import torch.nn as nn


def mse_loss(output, target):
    return nn.MSELoss()(output, target)