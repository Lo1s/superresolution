import torch.nn as nn


def mse_loss(output, target):
    loss = nn.MSELoss()
    return loss(output, target)