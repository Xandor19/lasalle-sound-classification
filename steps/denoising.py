import pywt
import librosa
import numpy as np
from globals.defaults import *
from scipy.signal import butter, sosfiltfilt, wiener
from sklearn.base import BaseEstimator, TransformerMixin


class BandpassFilter(BaseEstimator, TransformerMixin):
    """
    Bandpass filter to remove frequencies outside the specified range.

    ## Params:
    - low: Lower frequency threshold (Hz).
    - high: Upper frequency threshold (Hz).
    - order: Order of the filter.
    - fs: Sampling frequency of the waveform."""
    def __init__(self, low=20, high=1000, order=5, fs=SAMPLING_RATE):
        self.low = low
        self.high = high
        self.order = order
        self.fs = fs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nyq = 0.5 * self.fs
        # Get the filter coefficients
        b, a = butter(self.order, [self.low/nyq, self.high/nyq], btype='band', output='sos')

        # Apply the bidirectional filter to each signal in the batch
        return [sosfiltfilt(b, a, x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)) for x in X]


class SpectralSubtractor(BaseEstimator, TransformerMixin):
    """
    Spectral subtraction denoiser that estimates the noise profile from the input audio and subtracts it from the spectrogram.
    
    ## Params:
    - n_fft: Size of the FFT window.
    - hop_length: Number of samples between successive frames.
    - noise_percentile: Percentile to use for estimating the noise profile.
    """
    def __init__(self, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, noise_percentile=10):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_percentile = noise_percentile
        self.noise_profile_ = None

    def fit(self, X, y=None):
        all_spectra = []

        for x in X:
            # Safeguard against possible NaN values from previous steps
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # Get signal spectrum from STFT
            S = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length))
            all_spectra.append(S)

        # Wide concat of frequency spectrums from all signals
        combined = np.hstack(all_spectra)
        self.noise_profile_ = np.percentile(combined, self.noise_percentile, axis=1)

        return self

    def transform(self, X):
        cleaned = []

        for x in X:
            # Safeguard against possible NaN values from previous steps
            na_indices = np.isnan(x)

            if np.any(na_indices):
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
            # Compute STFT for the signal
            S = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude, phase = np.abs(S), np.angle(S)
            # Subtract noise estimate
            cleaned_mag = np.maximum(magnitude - self.noise_profile_[:, np.newaxis], 0)
            # Reconstruct the signal
            new_waveform = librosa.istft(cleaned_mag * np.exp(1j * phase), hop_length=self.hop_length)

            if np.any(na_indices):
                new_waveform[na_indices] = np.nan

            cleaned.append(new_waveform)

        return cleaned


class WienerFilter(BaseEstimator, TransformerMixin):
    """
    Wiener filter for signal denoising.
    
    ## Params:
    - mysize: Size of the filter window. Must be an odd number, defaults to 15.
    """
    def __init__(self, mysize=15):
        self.mysize = mysize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [wiener(x, mysize=self.mysize) for x in X]


class WaveletDenoiser(BaseEstimator, TransformerMixin):
    """
    Wavelet-based denoising strategy that uses thresholding to remove noise from the signal.
    
    ## Params:
    - wavelet: Type of wavelet to use for decomposition.
    - level: Level of wavelet decomposition.
    - threshold: Threshold value for wavelet coefficients.
    - mode: Thresholding mode, either "soft" or "hard".
    - substitute: Value to use for hard thresholding.
    """
    def __init__(self, wavelet=WAVELET_TYPE, level=4, threshold=0.02, mode="soft", substitute=0):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.mode = mode
        self.substitute = substitute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []

        for x in X:
            # Safeguard against possible NaN values from previous steps
            na_indices = np.isnan(x)

            if np.any(na_indices):
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Perform wavelet decomposition and extract threshold
            coeffs = pywt.wavedec(x, self.wavelet, level=self.level)
            coeffs = [pywt.threshold(c, self.threshold, mode=self.mode, substitute=self.substitute) for c in coeffs]
            new_waveform = pywt.waverec(coeffs, self.wavelet)

            if np.any(na_indices):
                new_waveform[na_indices] = np.nan

            cleaned.append(new_waveform)

        return cleaned


catalog = {
    "bandpass": BandpassFilter,
    "spectralsubs": SpectralSubtractor,
    "wiener": WienerFilter,
    "wavelet": WaveletDenoiser
}