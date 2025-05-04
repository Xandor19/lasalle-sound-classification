import torch
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T


class SpectrogramAugmenter(torch.nn.Module):
    """
    Applies transformations to spectrogram-like tensors for data augmentation.

    This class uses SpecMasker to perform time and frequency masking

    ** Params
    - freq_mask_param: Size of the masking in the frequency axis
    - time_mask_param: Size of the masking in the time axis
    - num_freq_masks: Number of masks to apply in the frequency axis
    - num_time_masks: Number of masks to apply in the time axis
    - noise_std: Standard deviation to use when adding Gaussian noise to the spectrogram. Zero to skip the transformation
    - time_shift_fraction: Size of the time segment to apply shifting to the spectrogram. Zero to skip the transformation
    - gain_scale: Multiplication factor to adjust gain. Zero to skip the transformation
    """
    def __init__(self, 
                 freq_mask_param=15, time_mask_param=30,
                 num_freq_masks=2, num_time_masks=2,
                 noise_std=0.01,
                 time_shift_fraction=0.1,
                 gain_scale=0.1):
        super().__init__()

        self.specmasker = SpecMasker(freq_mask_param, time_mask_param, num_freq_masks, num_time_masks)
        self.noise_std = noise_std
        self.time_shift_fraction = time_shift_fraction
        self.gain_scale = gain_scale

    def forward(self, spec_batch):
        B = spec_batch.shape[0]
        augmented = []

        for i in range(B):
            spec = spec_batch[i]

            # Brightness scaling
            if self.gain_scale > 0:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.gain_scale
                spec = spec * factor

            # Additive Gaussian noise
            if self.noise_std > 0:
                noise = torch.randn_like(spec) * self.noise_std
                spec = spec + noise

            # Time shifting
            if self.time_shift_fraction > 0:
                T_len = spec.shape[-1]
                max_shift = int(T_len * self.time_shift_fraction)
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                spec = torch.roll(spec, shifts=shift, dims=2)

            # Apply frequency/time masking
            augmented.append(self.specmasker(spec))

        return torch.stack(augmented)


class SpecMasker(torch.nn.Module):
    """
    Applies maskings to both frequency and time axes of an spectrogram-like tensor

    ** Params
    - freq_mask_param: Size of the masking in the frequency axis
    - time_mask_param: Size of the masking in the time axis
    - num_freq_masks: Number of masks to apply in the frequency axis
    - num_time_masks: Number of masks to apply in the time axis
    """
    def __init__(self, freq_mask_param=15, time_mask_param=30, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec):
        # Assume `spec` is a single [C, F, T] tensor
        if isinstance(spec, torch.Tensor):
            return self._apply_mask(spec)
        else:
            return torch.stack([self._apply_mask(spec[i]) for i in range(spec.shape[0])])


    def _apply_mask(self, spec):
        """
        Applies the stacked masking transforms
        """
        for _ in range(self.num_freq_masks):
            spec = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(spec)
        for _ in range(self.num_time_masks):
            spec = T.TimeMasking(time_mask_param=self.time_mask_param)(spec)

        return spec