"""Public-dataset loaders. Each `iter_<structure>()` yields
(meta: dict, x: float64[WINDOW] at 100 Hz) for non-overlapping windows.

meta keys: structure, file, channel, t0 (ISO), period, label (0 normal /
1 changed-or-damaged condition), temp (deg C or ''), split_key (unit that must
stay together when splitting; the source file).

`build_cache(structure, limit_per_file, version=)` turns the stream into
cache[/<version>]/<structure>.npz (float16 spectrograms) + .csv. Iterators take a
`frontend.FrontEnd` (sampling rate, window length, channel subset).
"""
import csv
import datetime as dt
import glob
import os
import zipfile

import numpy as np
import scipy.io as sio

from . import frontend as fe_mod

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, 'raw')
CACHE = os.path.join(HERE, 'cache')      # v1 cache dir; v<n> caches live in cache/v<n>/

# ---------------------------------------------------------------- Vänersborg
# Leander et al. 2023, Data in Brief 51:109729; Zenodo 10.5281/zenodo.8300495.
# 200 Hz, A1..A5 = columns 20..24 (0-based) in m/s^2. The crack propagated during
# the opening recorded in 2023_03_09T23_45_25.csv; that file is excluded.
VB_FS = 200
VB_ACC_COLS = list(range(20, 25))
VB_EVENT_FILE = '2023_03_09T23_45_25.csv'


def iter_vanersborg(fe=fe_mod.V1):
    files = sorted(glob.glob(os.path.join(RAW, 'vanersborg', '*.csv')))
    for f in files:
        name = os.path.basename(f)
        if name == VB_EVENT_FILE:
            continue
        t_file = dt.datetime.strptime(name[:19], '%Y_%m_%dT%H_%M_%S')
        label = 1 if t_file > dt.datetime(2023, 3, 9, 23, 45) else 0
        period = 'post-fracture (10-11 Mar 2023)' if label else 'pre-fracture (20 Feb - 9 Mar 2023)'
        a = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=VB_ACC_COLS + [32])
        temp = float(np.median(a[:, -1]))
        for ci, col in enumerate(VB_ACC_COLS):
            x = a[:, ci]
            if (x < -1e5).any():            # -1e6 is the DAQ's missing-value sentinel
                continue
            x = fe.resample(x, VB_FS)
            for s, w in fe.windows(x):
                yield dict(structure='VANERSBORG', file=name, channel='A%d' % (ci + 1),
                           t0=(t_file + dt.timedelta(seconds=s / fe.fs)).isoformat(), period=period,
                           label=label, temp='%.1f' % temp, split_key=name), w


# ------------------------------------------------------------------- Chillon
# Martín-Sanz et al. 2019, Struct. Infrastruct. Eng.; Zenodo 10.5281/zenodo.3234805.
# 100 Hz, 11 vertical accelerometers, 10-min files, air temperature T, humidity H.
# Strengthened viaduct under normal operation: every window is label 0.
def iter_chillon(fe=fe_mod.V1):
    files = sorted(glob.glob(os.path.join(RAW, 'chillon', '*.mat')))
    for f in files:
        name = os.path.basename(f)
        m = sio.loadmat(f)
        fs_in = int(m['Fs'])
        t_file = dt.datetime.strptime(name[:16], '%Y-%m-%d_%H-%M')
        data = m['dataZ']
        for ci in range(data.shape[1]):
            x = data[:, ci].astype(np.float64)
            if not np.isfinite(x).all() or x.std() == 0:
                continue
            x = fe.resample(x, fs_in)
            for s, w in fe.windows(x):
                yield dict(structure='CHILLON', file=name, channel='Z%d' % (ci + 1),
                           t0=(t_file + dt.timedelta(seconds=s / fe.fs)).isoformat(),
                           period='normal operation (Mar-Jun 2017)', label=0,
                           temp='%.1f' % float(m['T']), split_key=name), w


# ---------------------------------------------------------------------- KW51
# Maes & Lombaert 2021, J. Bridge Eng.; Zenodo 10.5281/zenodo.3745914.
# Ambient-vibration .mat files, struct predat_a with fields tdata (N x ch), fs,
# sdn (MATLAB datenum), labels. Retrofit 15 May - 27 Sep 2019.
# Reference (label 0) = post-retrofit, sound connections (Jan 2020).
# Changed (label 1) = pre-retrofit, deficient diagonal connections (Jan 2019).
KW51_MONTHS = {'202001': 0, '201901': 1}


def _kw51_files():
    out = []
    for month, label in KW51_MONTHS.items():
        z = os.path.join(RAW, 'kw51_ambient_%s.zip' % month)
        if not os.path.exists(z) or not zipfile.is_zipfile(z):   # absent or still downloading
            continue
        with zipfile.ZipFile(z) as zf:
            for n in sorted(zf.namelist()):
                if n.endswith('.mat'):
                    out.append((z, n, label))
    return out


def _datenum(x):
    return dt.datetime.fromordinal(int(x) - 366) + dt.timedelta(days=float(x) % 1)


def _kw51_env():
    """Hourly bridge-deck temperature tBD31A from trackedmodes.mat -> (sdn, temp)."""
    f = os.path.join(RAW, 'kw51_meta', 'trackedmodes', 'trackedmodes.mat')
    if not os.path.exists(f):
        return None
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)['modes']
    labels = list(np.atleast_1d(m.labels_env))
    return np.asarray(m.sdn, dtype=np.float64), np.asarray(m.env)[:, labels.index('tBD31A')]


def iter_kw51(fe=fe_mod.V1):
    import io
    env = _kw51_env()
    for z, n, label in _kw51_files():
        with zipfile.ZipFile(z) as zf:
            m = sio.loadmat(io.BytesIO(zf.read(n)), squeeze_me=True, struct_as_record=False)
        p = m['predat_a']
        fs = float(p.fs)
        data = np.atleast_2d(np.asarray(p.tdata, dtype=np.float64))
        if data.shape[0] < data.shape[1]:
            data = data.T
        labels = list(np.atleast_1d(p.labels))
        sdn0 = float(np.atleast_1d(p.sdn)[0])
        t_file = _datenum(sdn0)
        name = os.path.basename(n)
        period = 'pre-retrofit (Jan 2019)' if label else 'post-retrofit (Jan 2020)'
        temp = ''
        if env is not None:
            j = int(np.argmin(np.abs(env[0] - sdn0)))
            if abs(env[0][j] - sdn0) < 1 / 24 and np.isfinite(env[1][j]):
                temp = '%.1f' % env[1][j]
        for ci in range(data.shape[1]):
            if fe.channels and str(labels[ci]) not in fe.channels:
                continue
            x = data[:, ci]
            if not np.isfinite(x).all() or x.std() == 0:
                continue
            x = fe.resample(x, fs)
            for s, w in fe.windows(x):
                yield dict(structure='KW51', file=name, channel=str(labels[ci]),
                           t0=(t_file + dt.timedelta(seconds=s / fe.fs)).isoformat(), period=period,
                           label=label, temp=temp, split_key=name), w


ITER = {'VANERSBORG': iter_vanersborg, 'CHILLON': iter_chillon, 'KW51': iter_kw51}
META_COLS = ['structure', 'file', 'channel', 't0', 'period', 'label', 'temp', 'split_key']


def cache_dir(version):
    return CACHE if version == 'v1' else os.path.join(CACHE, version)


def build_cache(structure, limit_per_file=None, seed=0, version='v1'):
    """Spectrogram every window (optionally at most limit_per_file per source
    file, chosen uniformly) and store <cache_dir>/<structure>.{npz,csv}."""
    fe = fe_mod.get(version, structure)
    cdir = cache_dir(version)
    os.makedirs(cdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    specs, metas = [], []
    pending = []

    def flush():
        if limit_per_file and len(pending) > limit_per_file:
            keep = sorted(rng.choice(len(pending), limit_per_file, replace=False))
            sel = [pending[i] for i in keep]
        else:
            sel = pending
        for meta, w in sel:
            specs.append(fe.spectrogram(w).astype(np.float16)); metas.append(meta)
        pending.clear()

    cur = None
    for meta, w in ITER[structure](fe):
        if meta['file'] != cur:
            flush(); cur = meta['file']
        pending.append((meta, w))
    flush()
    arr = np.stack(specs) if specs else np.zeros((0,) + fe.img, np.float16)
    np.savez_compressed(os.path.join(cdir, structure + '.npz'), specs=arr)
    with open(os.path.join(cdir, structure + '.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=META_COLS); wr.writeheader()
        for m in metas:
            wr.writerow({k: m[k] for k in META_COLS})
    return arr.shape[0]


def load_cache(structure, version='v1'):
    cdir = cache_dir(version)
    specs = np.load(os.path.join(cdir, structure + '.npz'))['specs']
    with open(os.path.join(cdir, structure + '.csv')) as fh:
        metas = list(csv.DictReader(fh))
    assert len(metas) == specs.shape[0]
    return specs, metas


if __name__ == '__main__':
    import sys, time
    s = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '0' else None
    ver = sys.argv[3] if len(sys.argv) > 3 else 'v1'
    t = time.time(); n = build_cache(s, lim, version=ver)
    print(s, ver, n, 'windows cached in %.0f s' % (time.time() - t))
