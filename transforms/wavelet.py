import numpy as np
import pywt
import torch
import torch.nn.functional as F
from config import WAVELET, SCALES, IMAGE_SIZE


def timeseries_to_scalogram(signal: np.ndarray):
    scales = np.arange(1, SCALES + 1)
    coef, _ = pywt.cwt(signal, scales, WAVELET)
    coef = np.abs(coef)

    coef = (coef - coef.min()) / (coef.max() - coef.min() + 1e-8)

    tensor = torch.tensor(coef).unsqueeze(0).float()
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return tensor