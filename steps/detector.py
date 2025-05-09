import numpy as np
from steps.outlier_fixer import OutlierMasker
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from base import EntityTransformer, BaseCustom


class BaseOutlierDetector(BaseCustom, TransformerMixin):
    """
    Base class for outlier detection. Implements template method pattern to perform outlier detection and fixing

    ## Params:
    - fixer: Outlier fixer instance that will be used to replace outlier sections detected in the waveform. As there is no
                sense in detection without fixing, OutlierMasker is defaulted when receiving EntityTransformer.
    """
    def __init__(self, fixer, na_tolerant=True):
        super().__init__(na_tolerant=na_tolerant)
        self.fixer = fixer if not isinstance(fixer, EntityTransformer) else OutlierMasker()

    def fit(self, X, y=None):
        """No-op for fitting, as outlier detection is audio-dependent and e.g stateless"""
        return self

    def set_params(self, **params):
        self.fixer.set_params(**params)
        return super().set_params(**params)

    def _apply_transform(self, x):
        """Applies the corresponding outlier detection method and the received fixer"""
        return self.fixer.fix(x, self._detect(x))

    def _detect(self, X):
        """Base method for outlier detection. Should only indicate outlier sections, not modify them"""
        raise NotImplementedError("Subclasses should implement this method.")


class ZScoreOutlierDetector(BaseOutlierDetector):
    """
    Z-Score based outlier detector. Signals sections as outliers when they exceed the defined threshold.

    ## Params:
    - fixer: Outlier fixer instance that will be used to replace outlier sections detected in the waveform.
    - z_thresh: Z-Score threshold for outlier detection. Defaults to 5.
    """
    def __init__(self, fixer=OutlierMasker(), z_thresh=5):
        super().__init__(fixer, na_tolerant=True)
        self.z_thresh = z_thresh

    def _detect(self, X):
        z = (X - np.nanmean(X)) / np.nanstd(X)
        return np.abs(z) > self.z_thresh


class EnergyOutlierDetector(BaseOutlierDetector):
    """
    Energy-based outlier detector. Signals sections as outliers when their energy is above the defined percentile.

    ## Params:
    - fixer: Outlier fixer instance that will be used to replace outlier sections detected in the waveform.
    - energy_percentile: Percentile threshold for outlier detection. Defaults to 0.9.
    - frame_size: Frame size for energy window calculation. Defaults to 1024.
    """
    def __init__(self, fixer=OutlierMasker(), energy_percentile=90, multiplier=1, frame_size=1024):
        super().__init__(fixer, na_tolerant=True)
        self.frame_size = frame_size
        self.energy_percentile = energy_percentile
        self.multiplier = multiplier

    def _detect(self, X):
        # Get the energy of the frames and the defined percentile
        energy = [
            np.mean(X[i:i + self.frame_size] ** 2)
            for i in range(0, len(X) - self.frame_size, self.frame_size)
        ]
        threshold = np.percentile(energy, self.energy_percentile) * self.multiplier
        mask = np.zeros_like(X, dtype=bool)

        for i, e in enumerate(energy):
            # Mark frames in which energy exceeds the threshold
            if e > threshold:
                idx = i * self.frame_size
                mask[idx:idx + self.frame_size] = True

        return mask


class FlatSegmentDetector(BaseOutlierDetector):
    """
    Flat segment outlier detector. Flags sections as outliers when their variance is below the defined threshold.

    ## Params:
    - fixer: Outlier fixer instance that will be used to replace outlier sections detected in the waveform.
    - threshold: Variance threshold to flag a section as outlier. Defaults to near-zero value (1e-6).
    - frame_size: Frame size for energy window calculation. Defaults to 1024.
    """
    def __init__(self, fixer=OutlierMasker(), threshold=1e-6, frame_size=2048):
        super().__init__(fixer, na_tolerant=True)
        self.threshold = threshold
        self.frame_size = frame_size

    def _detect(self, X):
        mask = np.zeros_like(X, dtype=bool)

        for i in range(0, len(X) - self.frame_size, self.frame_size):
            # Get the frame range and compute variance
            frame = X[i:i + self.frame_size]

            if np.var(frame) < self.threshold:
                mask[i:i + self.frame_size] = True

        return mask


class IQROutlierDetector(BaseOutlierDetector):
    """
    Interquartile Range (IQR) based outlier detector. Flags sections as outliers when they are beyond the IQR plus a multiplier range.
    
    ## Params
    - fixer: Outlier fixer instance that will be used to replace outlier sections detected in the waveform.
    - k: Multiplier for the IQR. Defaults to 1.5.
    """
    def __init__(self, fixer=OutlierMasker(), k=1.5):
        super().__init__(fixer, na_tolerant=True)
        self.k = k

    def _detect(self, X):
        q1 = np.nanpercentile(X, 25)
        q3 = np.nanpercentile(X, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.k * iqr
        upper_bound = q3 + self.k * iqr

        return np.where((X < lower_bound) | (X > upper_bound))


class ExtremeCasesDetector(Pipeline):
    """
    Complex detector that combines Z-Score based peaks detection and flat segment detector to flag both extreme events and dead zones.
    Implements a pipeline that reuses the two standalone detectors.

    ## Params:
    - fixer: General fixer for both cases. When specified, both zscore_fixer and flat_fixer are overwritten. Defaults to None
    - zscore_fixer: Outlier fixer instance that will be used to replace outlier sections detected by the Z-Score method.
    - flat_fixer: Outlier fixer instance that will be used to replace outlier sections detected by the flat segments detector method.
    - z_thresh: Z-Score threshold for outlier detection. Defaults to 5.
    - flat_thresh: Variance threshold to flag a section as outlier. Defaults to near-zero value (1e-6).
    """
    def __init__(self, fixer=None, zscore_fixer=OutlierMasker(), flat_fixer=OutlierMasker(), z_thresh=5, flat_thresh=1e-6, flat_frame_size=2048):
        self.fixer = fixer
        self.zscore_fixer = zscore_fixer
        self.flat_fixer = flat_fixer
        self.z_thresh = z_thresh
        self.flat_thresh = flat_thresh
        self.flat_frame_size = flat_frame_size

        if fixer:
            zscore_fixer = fixer
            flat_fixer = fixer

        super().__init__([
            ('flat', FlatSegmentDetector(fixer=flat_fixer, threshold=flat_thresh, frame_size=flat_frame_size)),
            ('zscore', ZScoreOutlierDetector(fixer=zscore_fixer, z_thresh=z_thresh))
        ])


catalog = {
    "zscore": ZScoreOutlierDetector,
    "energy": EnergyOutlierDetector,
    "deathzone": FlatSegmentDetector,
    "iqr": IQROutlierDetector,
    "twosided": ExtremeCasesDetector
}