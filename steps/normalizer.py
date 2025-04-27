import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RMSNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes the RMS of each signal to a reference value.
    
    ## Params:
    - ref_rms: Reference RMS value to normalize the signals to. Defaults to 0.1."""
    def __init__(self, ref_rms=0.1, **kwargs):
        self.ref_rms = ref_rms

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        normalized = []

        for x in X:
            rms = np.sqrt(np.nanmean(x ** 2))
            normalized.append(x * (self.ref_rms / rms) if rms > 0 else x)

        return normalized


class PeakNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes the signal using the higher peak value that it contains.
    """
    def __init__(self, **kwargs):
        # Placeholder constructor to support generic fixers calls
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        normalized = []

        for x in X:
            max_val = np.max(np.abs(x))
            normalized.append(x / max_val if max_val > 0 else x)

        return normalized


catalog = {
    "rms": RMSNormalizer,
    "peak": PeakNormalizer
}