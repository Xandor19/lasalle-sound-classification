import numpy as np
from base import BaseCustom
from scipy.interpolate import interp1d


class OutlierMasker(BaseCustom):
    """
    Masks outlier section with NaN values to discard them in further processing.
    """
    def __init__(self, **kwargs):
        # Placeholder constructor to support generic fixers calls
        super().__init__(na_tolerant=True)

    def fix(self, signal, mask):
        masked = signal.copy()
        masked[mask] = np.nan

        return masked


class LinearInterpolatorMasker(BaseCustom):
    """
    Outlier removal by applying linear interpolation to the affected sections
    """
    def __init__(self, **kwargs):
        # Placeholder constructor to support generic fixers calls
        super().__init__(na_tolerant=True)

    def fix(self, signal, mask):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        signal = signal.copy()
        indices = np.arange(len(signal))
        good = ~mask

        # Not enough points to interpolate
        if np.sum(good) < 2:
            return signal

        interpolated = np.interp(indices, indices[good], signal[good])
        signal[mask] = interpolated[mask]

        return signal


class PolynomialInterpolatorMasker(BaseCustom):
    """
    Outlier removal by applying n-order polynomial interpolation to the affected sections. 
    If there are no enough points around the outlier, it falls back to linear interpolation.

    ## Params:
    - order: Order of the polynomial to use for interpolation. Defaults to 2 (quadratic).
    """
    def __init__(self, order=2):
        super().__init__(na_tolerant=True)
        self.order = order

    def fix(self, signal, mask):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        signal = signal.copy()
        indices = np.arange(len(signal))
        good = ~mask

        # Not enough points to interpolate
        if np.sum(good) <= self.order:
            return signal

        try:
            interp_func = interp1d(indices[good], signal[good], kind=self.order, fill_value="extrapolate")
            signal[mask] = interp_func(indices[mask])
        except Exception:
            # Fallback to linear if polynomial fails
            signal[mask] = np.interp(indices[mask], indices[good], signal[good])

        return signal


catalog = {
    "mask": OutlierMasker,
    "lininterp": LinearInterpolatorMasker,
    "polyinterp": PolynomialInterpolatorMasker,
}