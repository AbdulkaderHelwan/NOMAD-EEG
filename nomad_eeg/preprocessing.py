"""Input preprocessing used for every dataset in the paper.

1. optional band-pass (and notch) filtering, signals in microvolts;
2. resampling to 200 Hz, the pre-training rate;
3. per-channel robust scaling: subtract the median, divide by the inter-quartile range,
   clip at +-20. Statistics come from the TRAINING split only and are then applied
   unchanged to validation and test data.

Channels must have a standard 10-20 / 10-05 name (or "A-B" for a bipolar derivation);
drop ECG, EMG and other non-scalp channels before calling the model.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FS = 200
CLIP = 20.0


def preprocess(x: np.ndarray, fs: float, bandpass: tuple[float, float] | None = None,
               notch: float | None = None) -> np.ndarray:
    """Filter and resample x [..., T] (microvolts, sampled at fs) to 200 Hz, float32."""
    x = np.asarray(x, dtype=np.float64)
    if bandpass or notch:
        import mne
        if notch:
            x = mne.filter.notch_filter(x, fs, notch, verbose="ERROR")
        if bandpass:
            lo, hi = bandpass
            x = mne.filter.filter_data(x, fs, lo, min(hi, fs / 2 - 1), verbose="ERROR")
    if fs != FS:
        r = Fraction(FS, int(round(fs))).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator, axis=-1)
    return x.astype(np.float32)


class RobustScaler:
    """Per-channel median / IQR scaling with clipping, fitted on training windows [N, E, T]."""

    def __init__(self, median=None, iqr=None):
        self.median = None if median is None else np.asarray(median, np.float32)
        self.iqr = None if iqr is None else np.asarray(iqr, np.float32)

    def fit(self, x: np.ndarray, max_windows: int = 5000, seed: int = 0) -> "RobustScaler":
        x = np.asarray(x, np.float32)
        if len(x) > max_windows:
            x = x[np.sort(np.random.default_rng(seed).choice(len(x), max_windows, replace=False))]
        flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
        self.median = np.median(flat, axis=1).astype(np.float32)
        q75, q25 = np.percentile(flat, [75, 25], axis=1)
        self.iqr = np.maximum(q75 - q25, 1e-6).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, np.float32) - self.median[:, None]) / self.iqr[:, None]
        return np.clip(z, -CLIP, CLIP).astype(np.float32)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"median": self.median.tolist(), "iqr": self.iqr.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> "RobustScaler":
        d = json.loads(Path(path).read_text())
        return cls(d["median"], d["iqr"])
