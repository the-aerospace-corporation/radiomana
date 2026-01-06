"""Transforms compatible with 2D log PSD tensors"""

from functools import singledispatchmethod
from typing import Any, Dict, Union

import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import v2


class LogNoise(v2.Transform):
    """
    Transform to add noise to a log-scaled PSD tensor.
    """

    def __init__(self, p: float = 1, noise_power_db: float = -90):
        """
        Parameters
        ----------
        p : float, default 1
            Probability of the psd being noised.
        """
        super().__init__()
        self.p = p
        self.noise_power_db = noise_power_db

    def forward(self, *inputs):
        """Consume inputs in torchvision v2 style."""
        if len(inputs) == 1:
            # check if single input is a tuple (psd, label)
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) == 2:
                psd, label = inputs[0]
                return self._apply(psd), label
            else:
                return self._apply(inputs[0])
        elif len(inputs) == 2:
            label, psd = inputs
            return label, self._apply(psd)
        else:
            return super().forward(*inputs)

    def _apply(self, log_psd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        log_psd : Tensor
            Log PSD to add noise to of shape (freq, time) or (bs, freq, time).

        Returns
        -------
        Tensor
            Noised log PSD of same shape as input.
        """
        if torch.rand(1).item() < self.p:
            noise_power_linear = 10 ** (self.noise_power_db / 10)
            # convert data from dB to linear
            psd_linear = 10 ** (log_psd / 10)
            # add noise in linear domain
            noise = torch.randn_like(psd_linear) * noise_power_linear**0.5
            noisy_psd_linear = psd_linear + noise
            # ensure no negative or zero values before log10
            noisy_psd_linear = torch.clamp(noisy_psd_linear, min=1e-12)
            log_psd = 10 * torch.log10(noisy_psd_linear)
        return log_psd

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_power_db={self.noise_power_db}, p={self.p})"


class RandomTimeCrop(v2.Transform):
    """
    Transform to randomly crop a fixed-width time segment from a PSD tensor.
    Crops to a fixed width with random offset to ensure consistent batch dimensions.
    """

    def __init__(self, crop_width: int = 211):
        """
        Parameters
        ----------
        crop_width : int, default 211
            Fixed width to crop to in time dimension.
        """
        super().__init__()
        self.crop_width = crop_width

    def forward(self, *inputs):
        """Consume inputs in torchvision v2 style."""
        if len(inputs) == 1:
            # check if single input is a tuple (psd, label)
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) == 2:
                psd, label = inputs[0]
                return self._apply(psd), label
            else:
                return self._apply(inputs[0])
        elif len(inputs) == 2:
            label, psd = inputs
            return label, self._apply(psd)
        else:
            return super().forward(*inputs)

    def _apply(self, log_psd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        log_psd : Tensor
            Log PSD tensor of shape (freq, time) or (bs, freq, time).

        Returns
        -------
        Tensor
            Cropped log PSD tensor with fixed width in time dimension.
        """
        # handle both 2D (freq, time) and 3D (bs, freq, time) tensors
        if len(log_psd.shape) == 2:
            # shape: (freq, time)
            time_dim = log_psd.shape[1]
            if time_dim <= self.crop_width:
                # if input is already smaller than crop width, return as-is
                return log_psd
            # randomly choose starting position
            max_start = time_dim - self.crop_width
            start = torch.randint(0, max_start + 1, (1,)).item()
            return log_psd[:, start : start + self.crop_width]
        elif len(log_psd.shape) == 3:
            # shape: (bs, freq, time)
            time_dim = log_psd.shape[2]
            if time_dim <= self.crop_width:
                # if input is already smaller than crop width, return as-is
                return log_psd
            # randomly choose starting position
            max_start = time_dim - self.crop_width
            start = torch.randint(0, max_start + 1, (1,)).item()
            return log_psd[:, :, start : start + self.crop_width]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {log_psd.shape}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_width={self.crop_width})"
