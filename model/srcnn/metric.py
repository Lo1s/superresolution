import math
import numpy as np


def psnr(output, target, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    img_diff = output - target
    rmse = math.sqrt(np.mean(img_diff ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR