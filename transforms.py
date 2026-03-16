
import numpy as np
import torch
from typing import List, Optional, Union, Tuple

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomAmplitudeScaling:
    """Randomly scales the amplitude of the signal."""
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x: Union[np.ndarray, torch.Tensor]):
        if np.random.rand() < self.p:
            scale = np.random.uniform(*self.scale_range)
            return x * scale
        return x

class GaussianNoise:
    """Adds Gaussian noise to the signal."""
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: Union[np.ndarray, torch.Tensor]):
        if np.random.rand() < self.p:
            noise = np.random.normal(self.mean, self.std, x.shape).astype(x.dtype)
            if isinstance(x, torch.Tensor):
                return x + torch.from_numpy(noise).to(x.device)
            return x + noise
        return x

class TimeMask:
    """Masks random segments of the signal (like SpecAugment but for time)."""
    def __init__(self, max_mask_ratio: float = 0.1, n_masks: int = 2, p: float = 0.5):
        self.max_mask_ratio = max_mask_ratio
        self.n_masks = n_masks
        self.p = p

    def __call__(self, x: Union[np.ndarray, torch.Tensor]):
        if np.random.rand() < self.p:
            is_tensor = isinstance(x, torch.Tensor)
            data = x.clone().numpy() if is_tensor else x.copy()
            
            n_samples = data.shape[-1]
            max_mask_len = int(n_samples * self.max_mask_ratio)
            
            for _ in range(self.n_masks):
                mask_len = np.random.randint(0, max_mask_len)
                start = np.random.randint(0, n_samples - mask_len)
                data[..., start:start+mask_len] = 0
                
            return torch.from_numpy(data).to(x.device) if is_tensor else data
        return x

class ChannelDropout:
    """Randomly sets some channels to zero."""
    def __init__(self, max_channels: int = 2, p: float = 0.5):
        self.max_channels = max_channels
        self.p = p

    def __call__(self, x: Union[np.ndarray, torch.Tensor]):
        if np.random.rand() < self.p:
            is_tensor = isinstance(x, torch.Tensor)
            data = x.clone().numpy() if is_tensor else x.copy()
            
            n_channels = data.shape[0]
            n_drop = np.random.randint(1, self.max_channels + 1)
            drop_indices = np.random.choice(n_channels, n_drop, replace=False)
            
            data[drop_indices, :] = 0
            
            return torch.from_numpy(data).to(x.device) if is_tensor else data
        return x

class TimeShift:
    """Randomly shifts the signal in time."""
    def __init__(self, max_shift_samples: int = 100, p: float = 0.5):
        self.max_shift_samples = max_shift_samples
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]):
        if np.random.rand() < self.p:
            is_tensor = isinstance(x, torch.Tensor)
            data = x.clone().numpy() if is_tensor else x.copy()
            
            shift = np.random.randint(-self.max_shift_samples, self.max_shift_samples)
            data = np.roll(data, shift, axis=-1)
            
            # Zero out the rolled part to avoid leakage from end to start
            if shift > 0:
                data[..., :shift] = 0
            elif shift < 0:
                data[..., shift:] = 0
                
            return torch.from_numpy(data).to(x.device) if is_tensor else data
        return x

def get_default_train_transforms(sfreq=200):
    return Compose([
        RandomAmplitudeScaling(p=0.5),
        GaussianNoise(std=2.0, p=0.3), # Assuming uV scale, 2uV noise is reasonable
        TimeMask(max_mask_ratio=0.1, p=0.3),
        ChannelDropout(max_channels=2, p=0.2),
        TimeShift(max_shift_samples=int(0.5*sfreq), p=0.3)
    ])

def get_default_val_transforms():
    return Compose([])
