import numpy as np
import pandas as pd
import pywt
import librosa
from base import BaseCustom
from globals.defaults import *
from scipy.signal import hilbert
from sklearn.base import TransformerMixin


# Maps each feature to the transform it requires
FEATURE_TRANSFORM_MAP = {
    "mfcc": "fft",
    "spectral_centroid": "fft",
    "rolloff": "fft",
    "flatness": "fft",
    "bandwidth": "fft",
    "spectral_flux": "fft",
    "energy": "wavelet",
    "entropy": "wavelet",
    "envelope": "hilbert",
    "inst_freq": "hilbert",
    "spectral_flux_env": "hilbert",
}


def aggregate_feature(feat_name, values, methods):
    """
    Performs aggregations on the feature values based on the specified methods.

    ## Params:
    - feat_name: Name of the feature to be aggregated.
    - values: Values of the feature.
    - methods: List of aggregation methods to be applied. If "*" is present, all methods are applied.

    ## Returns:
    - Dictionary with the aggregated values for the feature, one per key, indicating the method used (single
      element dictionary if no aggregation is needed).
    """
    if methods and len(values) > 1:
        # Aggregated feature
        if "*" in methods:
            methods = ["mean", "std", "min", "max", "median", "iqr"]

        result = {}

        for m in methods:
            sufix = f"_{m}"

            match m:
                case "mean":
                    val = np.nanmean(values)
                case "std":
                    val = np.nanstd(values)
                case "iqr":
                    q75, q25 = np.nanpercentile(values, [75, 25])
                    val = q75 - q25
                case "median":
                    val = np.nanmedian(values)
                case "min":
                    val = np.nanmin(values)
                case "max":
                    val = np.nanmax(values)
                case _:
                    val = values
                    sufix = ""

            # Register the feature name with the corresponding aggregation
            result[f"{feat_name}{sufix}"] = val
    else:
        # Single value features are directly returned
        result = {feat_name: values[0] if hasattr(values, '__getitem__') else values}

    return result


class FeatureExtractor(BaseCustom, TransformerMixin):
    """
    Feature factory that computes both time and frequency domain features from audio signals. Transform-specific parameters
    are provided

    ## Params:
    - feature_config: Dictionary with the features to compute as keys and a list of aggregation methods to apply to them as values. 
                      It can be None to compute all possible features and aggregations or contain an "aggregator" key to apply
                      the same aggregation methods to all features, or specify a subset with a "features" key containing just a list of names
    - fs: Sampling rate of the audio signal.
    - n_fft: Size of the STFT window.
    - hop_length: Number of samples between successive frames.
    - n_mfcc: Number of MFCC coefficients to compute.
    - wavelet: Type of wavelet to use for wavelet transform.
    - wavelet_level: Level of the wavelet transform.
    """
    def __init__(self, feature_config=None, fs=SAMPLING_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH,
                 n_mfcc=13, wavelet=WAVELET_TYPE, wavelet_level=WAVELET_LEVEL):
        # Maps each feature to the function that computes it
        self._feature_functions = {
            "zcr": self._compute_zcr,
            "rms": self._compute_rms,
            "mfcc": self._compute_mfcc,
            "spectral_centroid": self._compute_spectral_centroid,
            "rolloff": self._compute_rolloff,
            "flatness": self._compute_flatness,
            "bandwidth": self._compute_bandwidth,
            "spectral_flux_env": self._compute_spectral_flux_env,
            "energy": self._compute_energy,
            "entropy": self._compute_entropy,
            "envelope": self._compute_envelope,
            "inst_freq": self._compute_inst_freq,
            "spectral_flux": self._compute_spectral_flux
        }
        # Construct the final feature config to use in the computation
        if "aggregator" in feature_config.keys():
            features = feature_config["features"] if "features" in feature_config.keys() else self._feature_functions.keys()
            self.feature_config = { f: feature_config["aggregator"] for f in features}
        else:
            self.feature_config = feature_config if feature_config else { k: ["*"] for k in self._feature_functions.keys() }

        for feature in feature_config.get("exclude", []):
            del self.feature_config[feature]

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []
        transforms_needed = { FEATURE_TRANSFORM_MAP[feature] for feature in self.feature_config.keys() if feature in FEATURE_TRANSFORM_MAP.keys() }

        for i, x in enumerate(X):
            feats = {}
            # Cache all transforms needed to compute the set of features
            cache = self._compute_transforms(x, transforms_needed)

            for feature_name, aggregations in self.feature_config.items():
                # Compute the raw value of the corresponding feature
                raw_values, nested = self._feature_functions[feature_name](x, cache)

                if nested:
                    # Features that generate multi-dimensional shapes (like MFCC) are aggregated one by one
                    for i, val in enumerate(raw_values):
                        feats.update(aggregate_feature(feature_name + f"_{i}", val, aggregations))
                else:
                    feats.update(aggregate_feature(feature_name, raw_values, aggregations))

            results.append(feats)

        return pd.DataFrame(results)

    def _compute_transforms(self, x, needed):
        cache = {}
        x = np.nan_to_num(x, **self._na_fill(x))

        if "fft" in needed:
            cache["fft"] = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length))

        if "wavelet" in needed:
            cache["wavelet"] = pywt.wavedec(x, self.wavelet, level=self.wavelet_level)

        if "hilbert" in needed:
            analytic = hilbert(x)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) * self.fs / (2 * np.pi)
            cache["hilbert"] = {
                "analytic": analytic,
                "envelope": np.abs(analytic),
                "inst_freq": inst_freq
            }

        return cache

    """
    Feature computation methods
    """

    def _compute_zcr(self, x, cache):
        return ((x[:-1] * x[1:]) < 0).astype(float), False

    def _compute_rms(self, x, cache):
        return np.array([np.sqrt(np.nanmean(x**2))]), False

    def _compute_mfcc(self, x, cache):
        x = np.nan_to_num(x, **self._na_fill(x))
        mfcc = librosa.feature.mfcc(y=x, sr=self.fs, n_mfcc=self.n_mfcc)

        return mfcc, True

    def _compute_spectral_centroid(self, x, cache):
        return librosa.feature.spectral_centroid(S=cache["fft"], sr=self.fs)[0], False

    def _compute_rolloff(self, x, cache):
        return librosa.feature.spectral_rolloff(S=cache["fft"], sr=self.fs)[0], False

    def _compute_flatness(self, x, cache):
        return librosa.feature.spectral_flatness(S=cache["fft"])[0], False

    def _compute_bandwidth(self, x, cache):
        return librosa.feature.spectral_bandwidth(S=cache["fft"], sr=self.fs)[0], False
    
    def _compute_spectral_flux(self, x, cache):
        S = cache["fft"]
        diff = np.diff(S, axis=1)
        flux = np.nansum(diff**2, axis=0)  # one value per frame

        return flux, False

    def _compute_energy(self, x, cache):
        return np.array([np.sum(c**2) for c in cache["wavelet"]]), False

    def _compute_entropy(self, x, cache):
        entropies = []

        for c in cache["wavelet"]:
            p = c**2 / (np.sum(c**2) + 1e-12)
            entropies.append(-np.sum(p * np.log2(p + 1e-12)))

        return np.array(entropies), False

    def _compute_envelope(self, x, cache):
        return cache["hilbert"]["envelope"], False

    def _compute_inst_freq(self, x, cache):
        return cache["hilbert"]["inst_freq"], False
    
    def _compute_spectral_flux_env(self, x, cache):
        envelope = cache["hilbert"]["envelope"]
        diff = np.diff(envelope)
        flux = diff ** 2

        return flux, False