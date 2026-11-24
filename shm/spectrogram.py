"""Window -> 100x100 log-PSD spectrogram, 0-10 Hz.

Protocol constants are the ones the manuscript states; the frame/step pair is
chosen so that an 11.9 s window yields exactly 100 time frames and the
zero-padded FFT yields 0.1 Hz bins, of which the first 100 cover 0-10 Hz.
"""
import numpy as np
from scipy import signal

FS = 100            # Hz, every structure is resampled to this
WINDOW = 1190       # samples, 11.9 s
FRAME = 497         # samples per Hanning frame (4.97 s, 0.2 Hz true resolution)
STEP = 7            # samples between frames: 497 + 99*7 = 1190 -> 100 frames
NFFT = 1000         # 0.1 Hz bin spacing
NBINS = 100         # bins 0..99 -> 0.0 .. 9.9 Hz
IMG = 100


def spectrogram(x):
    """x: (WINDOW,) float array at FS. Returns (IMG, IMG) float32 log10 PSD
    with time on axis 0 and frequency on axis 1."""
    assert x.shape == (WINDOW,), x.shape
    x = x - x.mean()
    _, _, sxx = signal.spectrogram(x, FS, window='hann', nperseg=FRAME,
                                   noverlap=FRAME - STEP, nfft=NFFT,
                                   mode='psd', detrend=False)
    sxx = sxx[:NBINS, :]                       # (freq, time)
    assert sxx.shape == (NBINS, IMG), sxx.shape
    return np.log10(sxx + 1e-20).T.astype(np.float32)


def resample_to_fs(x, fs_in):
    """Resample a 1-D signal to FS with an anti-aliasing polyphase filter."""
    if fs_in == FS:
        return np.asarray(x, dtype=np.float64)
    from fractions import Fraction
    fr = Fraction(FS, int(round(fs_in))).limit_denominator(1000)
    return signal.resample_poly(np.asarray(x, dtype=np.float64), fr.numerator, fr.denominator)


def windows(x, step=WINDOW):
    """Yield (start_sample, window) for non-overlapping windows by default."""
    n = (len(x) - WINDOW) // step + 1
    for i in range(max(n, 0)):
        s = i * step
        yield s, x[s:s + WINDOW]


class MinMax:
    """Global min-max normalizer fitted on training spectrograms (as in the
    reference implementation); stored with the model."""

    def __init__(self, lo=None, hi=None):
        self.lo, self.hi = lo, hi

    def fit(self, specs):
        self.lo = float(np.min(specs)); self.hi = float(np.max(specs))
        return self

    def __call__(self, s):
        return (s - self.lo) / (self.hi - self.lo)

    def state(self):
        return {'lo': self.lo, 'hi': self.hi}
