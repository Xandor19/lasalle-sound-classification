import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseCustom(BaseEstimator):
    """
    Template class for all custom components that handles the update of parameters and the application
    of batch transformations
    """
    def __init__(self, na_tolerant=True):
        self.na_tolerant = na_tolerant

    def transform(self, X):
        transformed = []

        for x in X:
            if not self.na_tolerant:
                # Safeguard against possible NaN values from previous steps when the method is not na-tolerant
                na_indices = np.isnan(x)

                if np.any(na_indices): 
                    x = np.nan_to_num(x, **self._na_fill(x))

            new_waveform = self._apply_transform(x)

            if not self.na_tolerant and np.any(na_indices):
                # Reset na indices to their original value
                new_waveform[na_indices] = np.nan

            transformed.append(new_waveform)

        return transformed

    def set_params(self, **params):
        """Override to avoid failure when receiving more parameters than expected, simply ignoring them"""
        for p, v in params.items():
            try:
                super().set_params(**{p: v})
            except:
                pass

        return self
    
    def _na_fill(self, x):
        return {
            "nan": 0.0,
            "posinf": 0.0,
            "neginf": 0.0
        }

    def _apply_transform(self, x):
        """Base method for individual instances transforming"""
        raise NotImplementedError("Subclasses should implement this method.")


class EntityTransformer(BaseEstimator, TransformerMixin):
    """
    A fallback class to skip one pipeline step.
    """

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
