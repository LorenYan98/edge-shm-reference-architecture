"""Per-structure signal front ends: sampling rate, window length, spectrogram
geometry, channel subset and level normalisation. Every front end yields a
100 (time) x NBINS (frequency) log10-PSD image.

v1  (manuscript draft, one front end for all):  100 Hz, 11.9 s windows,
    0.1 Hz bins over 0-9.9 Hz, 100x100, absolute level kept.
v2  (after out/diag): per-window level removed (the v1 error tracked signal
    power, r = 0.89); KW51/Chillon 60 s windows with 0.05 Hz bins over
    0-9.95 Hz so the 0.06-0.2 Hz modal shifts of the KW51 retrofit span
    several bins; Vänersborg native 200 Hz, 0.5 Hz bins over 0-99.5 Hz
    because the fracture signature sits at 20-85 Hz; KW51 restricted to the
    six deck channels present in both years.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFRAMES = 100


@dataclass(frozen=True)
class FrontEnd:
    name: str
    fs: int            # Hz after resampling
    window: int        # samples per window
    frame: int         # Hanning frame length, samples
    step: int          # frame step, samples; frame + (NFRAMES-1)*step == window
    nfft: int          # zero-padded FFT length -> bin spacing fs/nfft
    nbins: int         # frequency bins kept, starting at 0 Hz
    patch: Tuple[int, int]            # (time, freq) patch size for the ViT
    level_norm: bool = False          # subtract the window's mean log-PSD
    channels: Optional[Tuple[str, ...]] = None   # restrict to these channel labels

    def __post_init__(self):
        assert self.frame + (NFRAMES - 1) * self.step == self.window, self.name
        assert self.nbins <= self.nfft // 2 + 1
        assert NFRAMES % self.patch[0] == 0 and self.nbins % self.patch[1] == 0

    @property
    def img(self):
        return (NFRAMES, self.nbins)

    @property
    def df(self):
        return self.fs / self.nfft

    @property
    def cache_dir(self):
        return os.path.join(HERE, 'cache', self.name)

    def spectrogram(self, x):
        """x: (window,) at fs. Returns (NFRAMES, nbins) float32 log10 PSD."""
        assert x.shape == (self.window,), x.shape
        x = x - x.mean()
        _, _, sxx = signal.spectrogram(x, self.fs, window='hann', nperseg=self.frame,
                                       noverlap=self.frame - self.step, nfft=self.nfft,
                                       mode='psd', detrend=False)
        sxx = sxx[:self.nbins, :]
        assert sxx.shape == (self.nbins, NFRAMES), sxx.shape
        s = np.log10(sxx + 1e-20).T
        if self.level_norm:
            s = s - s.mean()
        return s.astype(np.float32)

    def resample(self, x, fs_in):
        if int(round(fs_in)) == self.fs:
            return np.asarray(x, dtype=np.float64)
        from fractions import Fraction
        fr = Fraction(self.fs, int(round(fs_in))).limit_denominator(1000)
        return signal.resample_poly(np.asarray(x, dtype=np.float64), fr.numerator, fr.denominator)

    def windows(self, x):
        n = (len(x) - self.window) // self.window + 1
        for i in range(max(n, 0)):
            s = i * self.window
            yield s, x[s:s + self.window]


KW51_DECK = ('aBD11Az', 'aBD17Ay', 'aBD17Az', 'aBD17Cz', 'aBD23Ay', 'aBD23Az')

V1 = FrontEnd('v1', fs=100, window=1190, frame=497, step=7, nfft=1000, nbins=100, patch=(5, 5))
V2_LOW = FrontEnd('v2', fs=100, window=5960, frame=2000, step=40, nfft=2000, nbins=200, patch=(5, 10), level_norm=True)
V2_KW51 = FrontEnd('v2', fs=100, window=5960, frame=2000, step=40, nfft=2000, nbins=200, patch=(5, 10),
                   level_norm=True, channels=KW51_DECK)
V2_WIDE = FrontEnd('v2', fs=200, window=11983, frame=400, step=117, nfft=400, nbins=200, patch=(5, 10), level_norm=True)

FRONTENDS = {
    'v1': {'KW51': V1, 'VANERSBORG': V1, 'CHILLON': V1},
    'v2': {'KW51': V2_KW51, 'VANERSBORG': V2_WIDE, 'CHILLON': V2_LOW},
}


def get(version, structure):
    return FRONTENDS[version][structure]
