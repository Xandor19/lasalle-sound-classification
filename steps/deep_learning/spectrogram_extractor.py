import pywt
import torch
import numpy as np
import torchaudio.transforms as T
from base import BaseCustom
from globals.defaults import *
from sklearn.base import TransformerMixin


class SpectrogramExtractor(BaseCustom, TransformerMixin):
    """
    Transformer that converts raw audio waveforms into spectrogram-like tensors using a selected transform method. 
    Regions that were masked using np.nan will be masked with zeros on the final tensor

    ** Params:
    - transform_type: 'stft', 'mel', or 'wavelet'
    - transform_params: dictionary with the parameter names and corresponding values for each transform. Names
                        match the source libraries (e.g, torchaudio, pywt) parameter names for computing the transforms
    """
    def __init__(self, transform_type='stft', transform_params={}):
        super().__init__(na_tolerant=False, rebuild_transformed=False)
        self.transform_type = transform_type
        self.transform_params = transform_params
        self.requires_tensor = True
        self.transformer = self._init_transformer()

    def _init_transformer(self):
        """Initializes the transformer to the corresponding functionality"""
        if self.transform_type == 'stft':
            return T.Spectrogram(**self.transform_params)
        elif self.transform_type == 'mel':
            return torch.nn.Sequential(
                T.MelSpectrogram(**self.transform_params),
                T.AmplitudeToDB(stype='power')
            )
        elif self.transform_type == 'wavelet':
            self.requires_tensor = False
            return self._compute_wavelet
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

    def fit(self, X, y=None):
        return self

    def _apply_transform(self, x):
        if self.requires_tensor:
            x = torch.tensor(x, dtype=torch.float32) 
            
            if x.dim() == 1:
                x = x.unsqueeze(0)

        spec = self.transformer(x)
        spec = (spec - spec.min()) / (spec.max() - spec.min())

        # Apply temporal mask if NA indices are provided
        if self.na_indices is not None:
            spec = self._mask_invalid_time_frames(spec, len(x))

        return spec

    def _compute_wavelet(self, x):
        """
        Computes a continuous wavelet of the specified type and stacks it into an scalogram-like tensor
        """
        wavelet = self.transform_params.get("wavelet", CONTINUOUS_WAVELET)
        num_scales = self.transform_params.get("scales", 64)
        sr = self.transform_params.get("sampling_rate", SAMPLING_RATE)

        scales = np.arange(1, num_scales + 1)
        coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1/sr)

        return torch.tensor(coeffs, dtype=torch.float32).unsqueeze(0)

    def _mask_invalid_time_frames(self, spec, signal_length):
        """
        Zero out time frames in the spectrogram corresponding to regions
        where a NaN mask was applied in the waveform.
        """
        time_dim = spec.shape[-1]

        # Estimate frame boundaries
        mask = torch.zeros(time_dim, dtype=torch.float32)
        frame_indices = np.linspace(0, signal_length, time_dim + 1, endpoint=True).astype(int)

        for i in range(time_dim):
            start, end = frame_indices[i], frame_indices[i + 1]
            if np.mean(self.na_indices[start:end]) > 0.5:
                mask[i] = 0.0
            else:
                mask[i] = 1.0

        # Broadcast mask to mask spectrogram shape of [1, freq, time] and apply it
        mask = mask.view(1, 1, -1)
        spec = spec * mask

        return spec

    def set_params(self, **params):
        super().set_params(**params)
        self.transformer = self._init_transformer()

        return self