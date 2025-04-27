import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RMSNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes the RMS of each signal to a reference value.
    
    ## Params:
    - ref_rms: Reference RMS value to normalize the signals to. Defaults to 0.1."""
    def __init__(self, ref_rms=1, **kwargs):
        self.ref_rms = ref_rms

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        normalized = []

        for x in X:
            rms = np.sqrt(np.nanmean(x ** 2))
            normalized.append(x * (self.ref_rms / rms) if rms > 0 else x)

        return normalized
    

class StandardNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes the amplitudes of the signal so they follow standard distribution
    """
    def __init__(self, **kwargs):
        # Placeholder constructor to support generic fixers calls
        pass

    def fit(self, X, y=None):
        concatenated = np.concatenate(X)

        # Compute training mean and std
        self.train_mean = np.nanmean(concatenated)
        self.train_std = np.nanstd(concatenated)
        return self

    def transform(self, X):
        return [(x - self.train_mean) / self.train_std for x in X]


catalog = {
    "rms": RMSNormalizer,
    "standard": StandardNormalizer
}