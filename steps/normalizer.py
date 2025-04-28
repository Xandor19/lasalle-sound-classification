import numpy as np
from base import BaseCustom
from sklearn.base import TransformerMixin


class RMSNormalizer(BaseCustom, TransformerMixin):
    """
    Normalizes the RMS of each signal to a reference value.

    ## Params:
    - ref_rms: Reference RMS value to normalize the signals to. Defaults to 0.1."""
    def __init__(self, ref_rms=1):
        self.ref_rms = ref_rms

    def fit(self, X, y=None):
        return self

    def _apply_transform(self, x):
        rms = np.sqrt(np.nanmean(x ** 2))
        return (x * (self.ref_rms / rms) if rms > 0 else x)
  

class StandardNormalizer(BaseCustom, TransformerMixin):
    """
    Normalizes the amplitudes of the signal so they follow standard distribution
    """
    def __init__(self, **kwargs):
        self.train_mean = None
        self.train_std = None
        pass

    def fit(self, X, y=None):
        concatenated = np.concatenate(X)

        # Compute training mean and std
        self.train_mean = np.nanmean(concatenated)
        self.train_std = np.nanstd(concatenated)
        return self

    def _apply_transform(self, x):
        return (x - self.train_mean) / self.train_std


catalog = {
    "rms": RMSNormalizer,
    "standard": StandardNormalizer
}