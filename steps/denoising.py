import pywt
import librosa
import numpy as np
from base import BaseCustom
from globals.defaults import *
from scipy.signal import butter, sosfiltfilt, wiener
from sklearn.base import BaseEstimator, TransformerMixin


class BandpassFilter(BaseCustom, TransformerMixin):
    """
    Bandpass filter to remove frequencies outside the specified range.

    ## Params:
    - low: Lower frequency threshold (Hz).
    - high: Upper frequency threshold (Hz).
    - order: Order of the filter.
    - fs: Sampling frequency of the waveform."""
    def __init__(self, low=20, high=1000, order=5, fs=SAMPLING_RATE):
        super().__init__(na_tolerant=False)
        self.low = low
        self.high = high
        self.order = order
        self.fs = fs
        self.order_coeffs = None
        self._compute_coefficients()

    def fit(self, X, y=None):
        return self

    def _apply_transform(self, x):       
        # Apply the bidirectional filter to each signal in the batch
        return sosfiltfilt(self.order_coeffs, x)

    def set_params(self, **params):
        super().set_params(**params)
        self._compute_coefficients()

        return self

    def _compute_coefficients(self):
        nyq = 0.5 * self.fs
        self.order_coeffs = butter(self.order, [self.low/nyq, self.high/nyq], btype='band', output='sos')


class SpectralSubtractor(BaseCustom, TransformerMixin):
    """
    Spectral subtraction denoiser that estimates the noise profile from the input audio and subtracts it from the spectrogram.
    
    ## Params:
    - n_fft: Size of the FFT window.
    - hop_length: Number of samples between successive frames.
    - noise_percentile: Percentile to use for estimating the noise profile.
    """
    def __init__(self, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, noise_percentile=10):
        super().__init__(na_tolerant=False)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_percentile = noise_percentile
        self.noise_profile_ = None

    def fit(self, X, y=None):
        all_spectra = []

        for x in X:
            # Safeguard against possible NaN values from previous steps
            x = np.nan_to_num(x, **self._na_fill(x))
            # Get signal spectrum from STFT
            S = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length))
            all_spectra.append(S)

        # Wide concat of frequency spectrums from all signals
        combined = np.hstack(all_spectra)
        self.noise_profile_ = np.percentile(combined, self.noise_percentile, axis=1)

        return self

    def _apply_transform(self, x):
        # Compute STFT for the signal
        S = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = np.abs(S), np.angle(S)
        # Subtract noise estimate
        cleaned_mag = np.maximum(magnitude - self.noise_profile_[:, np.newaxis], 0)
        # Reconstruct the signal
        new_waveform = librosa.istft(cleaned_mag * np.exp(1j * phase), hop_length=self.hop_length)

        return new_waveform


class WienerFilter(BaseCustom, TransformerMixin):
    """
    Wiener filter for signal denoising.
    
    ## Params:
    - mysize: Size of the filter window. Must be an odd number, defaults to 15.
    """
    def __init__(self, mysize=15):
        super().__init__(na_tolerant=False)
        self.mysize = mysize

    def fit(self, X, y=None):
        return self

    def _apply_transform(self, x):
        return wiener(x, mysize=self.mysize)


class WaveletDenoiser(BaseCustom, TransformerMixin):
    """
    Wavelet-based denoising strategy that uses thresholding to remove noise from the signal.
    
    ## Params:
    - wavelet: Type of wavelet to use for decomposition.
    - level: Level of wavelet decomposition.
    - threshold: Threshold value for wavelet coefficients.
    - mode: Thresholding mode, either "soft" or "hard".
    - substitute: Value to use for hard thresholding.
    """
    def __init__(self, wavelet=WAVELET_TYPE, level=WAVELET_LEVEL, threshold=0.02, mode="soft", substitute=0):
        super().__init__(na_tolerant=False)
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.mode = mode
        self.substitute = substitute

    def fit(self, X, y=None):
        return self

    def _apply_transform(self, x):
        # Perform wavelet decomposition and extract threshold
        coeffs = pywt.wavedec(x, self.wavelet, level=self.level)
        coeffs = [pywt.threshold(c, self.threshold, mode=self.mode, substitute=self.substitute) for c in coeffs]
        new_waveform = pywt.waverec(coeffs, self.wavelet)

        return new_waveform


catalog = {
    "bandpass": BandpassFilter,
    "spectralsubs": SpectralSubtractor,
    "wiener": WienerFilter,
    "wavelet": WaveletDenoiser
}