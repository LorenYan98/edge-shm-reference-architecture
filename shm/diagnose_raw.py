"""Raw-data diagnostics (no training): is the condition change visible in the
INPUT if we change the front end?

KW51: per-file Welch PSD at 0.02 Hz on the 6 deck channels present in both
years, level-normalised. Reports AUC of a mean-spectrum z^2 detector at 0.02 Hz
vs 0.1 Hz, and the frequency bins that differ most between years.
Vänersborg: per-opening Welch PSD 0-100 Hz (native 200 Hz), level-normalised.
Reports band-wise AUC (0-10, 10-30, 30-60, 60-100 Hz) pre vs post fracture.
Writes out/diag/kw51_hires.csv, vanersborg_bands.csv.
"""
import csv
import glob
import io
import os
import zipfile

import numpy as np
import scipy.io as sio
from scipy import signal

from .datasets import RAW, VB_ACC_COLS, VB_EVENT_FILE, _kw51_files, resample_to_fs
from .diagnose import DIAG, auc

KW_SHARED = ['aBD11Az', 'aBD17Ay', 'aBD17Az', 'aBD17Cz', 'aBD23Ay', 'aBD23Az']


def welch_log(x, fs, df):
    f, p = signal.welch(x - x.mean(), fs, nperseg=int(round(fs / df)), window='hann')
    return f, np.log10(p + 1e-20)


def z2_auc(P, y, train_frac=0.6, seed=0):
    """Fit per-bin mean/sd on a random 60 % of label-0 rows; score all others."""
    rng = np.random.RandomState(seed)
    i0 = np.where(y == 0)[0]; rng.shuffle(i0)
    tr = i0[:int(train_frac * len(i0))]
    rest = np.setdiff1d(np.arange(len(y)), tr)
    mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-6
    d = (((P[rest] - mu) / sd) ** 2).mean(1)
    return auc(y[rest], d)


def kw51():
    rows, P02, P10, Y, F02 = [], [], [], [], None
    for z, n, label in _kw51_files():
        with zipfile.ZipFile(z) as zf:
            m = sio.loadmat(io.BytesIO(zf.read(n)), squeeze_me=True, struct_as_record=False)['predat_a']
        data = np.atleast_2d(np.asarray(m.tdata, dtype=np.float64))
        if data.shape[0] < data.shape[1]:
            data = data.T
        labels = list(np.atleast_1d(m.labels))
        for c in KW_SHARED:
            if c not in labels:
                continue
            x = resample_to_fs(data[:, labels.index(c)], float(m.fs))
            if not np.isfinite(x).all() or x.std() == 0:
                continue
            f, p = welch_log(x, 100, 0.02); sel = f <= 10
            f1, p1 = welch_log(x, 100, 0.1); sel1 = f1 <= 10
            P02.append(p[sel] - p[sel].mean()); P10.append(p1[sel1] - p1[sel1].mean()); Y.append(label)
            F02 = f[sel]
    P02, P10, Y = np.array(P02), np.array(P10), np.array(Y)
    out = dict(n_ref=int((Y == 0).sum()), n_chg=int((Y == 1).sum()),
               auc_z2_0p1Hz=z2_auc(P10, Y), auc_z2_0p02Hz=z2_auc(P02, Y))
    # which bins move: Cohen's d per 0.02 Hz bin
    d = (P02[Y == 1].mean(0) - P02[Y == 0].mean(0)) / (0.5 * (P02[Y == 1].std(0) + P02[Y == 0].std(0)) + 1e-6)
    top = np.argsort(-np.abs(d))[:12]
    out['top_bins_Hz_cohen_d'] = [(round(float(F02[i]), 2), round(float(d[i]), 2)) for i in sorted(top)]
    with open(os.path.join(DIAG, 'kw51_hires.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['freq', 'ref_mean', 'chg_mean', 'cohen_d'])
        for i in range(len(F02)):
            w.writerow(['%.2f' % F02[i], '%.4f' % P02[Y == 0, i].mean(), '%.4f' % P02[Y == 1, i].mean(), '%.3f' % d[i]])
    return out


def vanersborg():
    bands = [(0, 10), (10, 30), (30, 60), (60, 100)]
    P, Y, F = [], [], None
    for fpath in sorted(glob.glob(os.path.join(RAW, 'vanersborg', '*.csv'))):
        name = os.path.basename(fpath)
        if name == VB_EVENT_FILE:
            continue
        label = 1 if name[:19] > '2023_03_09T23_45_25' else 0
        a = np.genfromtxt(fpath, delimiter=',', skip_header=1, usecols=VB_ACC_COLS)
        for ci in range(a.shape[1]):
            x = a[:, ci]
            if (x < -1e5).any():
                continue
            f, p = welch_log(x, 200, 0.1)
            P.append(p - p.mean()); Y.append(label); F = f
    P, Y = np.array(P), np.array(Y)
    out = dict(n_ref=int((Y == 0).sum()), n_chg=int((Y == 1).sum()))
    with open(os.path.join(DIAG, 'vanersborg_bands.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['band_Hz', 'auc_z2'])
        for lo, hi in bands:
            sel = (F >= lo) & (F < hi)
            a_ = z2_auc(P[:, sel], Y); out['auc_z2_%d_%dHz' % (lo, hi)] = a_
            w.writerow(['%d-%d' % (lo, hi), '%.3f' % a_])
    out['auc_z2_0_100Hz'] = z2_auc(P, Y)
    d = (P[Y == 1].mean(0) - P[Y == 0].mean(0)) / (0.5 * (P[Y == 1].std(0) + P[Y == 0].std(0)) + 1e-6)
    top = np.argsort(-np.abs(d))[:12]
    out['top_bins_Hz_cohen_d'] = [(round(float(F[i]), 1), round(float(d[i]), 2)) for i in sorted(top)]
    return out


if __name__ == '__main__':
    import json
    os.makedirs(DIAG, exist_ok=True)
    r = {'KW51': kw51(), 'VANERSBORG': vanersborg()}
    print(json.dumps(r, indent=1))
    json.dump(r, open(os.path.join(DIAG, 'report_raw.json'), 'w'), indent=1)
