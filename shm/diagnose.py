"""No-training diagnostics on cached spectrograms + saved scores.

Answers, per structure:
  1. Is the condition change visible in the INPUT at all? (mean log-PSD per
     label, peak shifts, total-power difference, a trivial mean-spectrum
     distance detector and its AUC)
  2. Where does the model fail? (AUC per channel / per file, score drift
     train -> calib -> test on reference data, inverted-AUC check)
  3. Confounds: temperature, amplitude, channel set per label.
Writes out/diag/<structure>_*.csv and prints a compact report.
"""
import csv
import json
import os

import numpy as np

from .train import RUNS, STRUCTURES, load_all

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(HERE, 'out', 'diag')
FREQ = np.arange(100) * 0.1


def auc(y, e):
    """Rank AUC of e for y==1 vs y==0 (nan if one class missing)."""
    y = np.asarray(y); e = np.asarray(e, dtype=np.float64)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = e.argsort().argsort().astype(np.float64) + 1   # ranks (ties ignored)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def peaks(spec_mean, k=6, lo=0.3):
    """Top-k local maxima of a mean spectrum (freq, value), above lo Hz."""
    s = spec_mean
    idx = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and FREQ[i] >= lo]
    idx = sorted(idx, key=lambda i: -s[i])[:k]
    return sorted((FREQ[i], float(s[i])) for i in idx)


def main():
    os.makedirs(DIAG, exist_ok=True)
    run = os.path.join(RUNS, 'mae_v1')
    cfg = json.load(open(os.path.join(run, 'config.json')))
    data = load_all(seed=cfg['split_seed'])
    report = {}
    for s, specs, metas, split in data:
        z = np.load(os.path.join(run, 'scores_%s.npz' % s)); ev, ep = z['vit'], z['pca']
        split = np.array(split); y = np.array([int(m['label']) for m in metas])
        ch = np.array([m['channel'] for m in metas]); fl = np.array([m['file'] for m in metas])
        temp = np.array([float(m['temp']) if m['temp'] else np.nan for m in metas])
        test = split == 'test'
        R = {}

        # --- 1. input-level separability ---------------------------------
        X = specs.astype(np.float32)                       # (n,100 time,100 freq)
        mean_spec = X.mean(1)                              # (n, 100 freq) time-avg log PSD
        power = mean_spec.mean(1)                          # proxy for log total power
        ref_tr = (split == 'train')
        mu = mean_spec[ref_tr].mean(0); sd = mean_spec[ref_tr].std(0) + 1e-6
        dist = (((mean_spec - mu) / sd) ** 2).mean(1)      # per-bin z^2, averaged
        R['auc_vit_test'] = auc(y[test], ev[test])
        R['auc_pca_test'] = auc(y[test], ep[test])
        R['auc_meanspec_z2_test'] = auc(y[test], dist[test])
        R['auc_power_test'] = auc(y[test], power[test])
        R['auc_power_inv_test'] = auc(y[test], -power[test])
        R['power_mean_by_label'] = {int(l): float(power[y == l].mean()) for l in np.unique(y)}
        R['peaks_by_label'] = {int(l): peaks(mean_spec[y == l].mean(0)) for l in np.unique(y)}
        with open(os.path.join(DIAG, '%s_meanspec.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['freq'] + ['label%d' % l for l in np.unique(y)])
            for i in range(100):
                w.writerow(['%.1f' % FREQ[i]] + ['%.4f' % mean_spec[y == l, i].mean() for l in np.unique(y)])

        # --- 2. where the model fails -------------------------------------
        R['vit_err_median_by_split_ref'] = {sp: float(np.median(ev[(split == sp) & (y == 0)]))
                                           for sp in ('train', 'calib', 'test')}
        R['vit_err_median_by_label_test'] = {int(l): float(np.median(ev[test & (y == l)])) for l in np.unique(y)}
        R['channels_by_label'] = {int(l): sorted(set(ch[y == l])) for l in np.unique(y)}
        rows = []
        for c in sorted(set(ch)):
            m = test & (ch == c)
            rows.append([c, int(m.sum()), int((y[m] == 1).sum()), auc(y[m], ev[m]), auc(y[m], ep[m]),
                         auc(y[m], dist[m]), float(np.median(ev[m & (y == 0)])) if (m & (y == 0)).any() else np.nan,
                         float(np.median(ev[m & (y == 1)])) if (m & (y == 1)).any() else np.nan])
        with open(os.path.join(DIAG, '%s_channels.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['channel', 'n_test', 'n_pos', 'auc_vit', 'auc_pca', 'auc_meanspec', 'med_vit_ref', 'med_vit_chg'])
            for r in rows:
                w.writerow([r[0], r[1], r[2]] + ['%.3f' % v if np.isfinite(v) else '' for v in r[3:6]] + ['%.5g' % v if np.isfinite(v) else '' for v in r[6:]])
        R['channel_auc_vit'] = {r[0]: round(r[3], 3) for r in rows if np.isfinite(r[3])}
        # per-file means: does the file-level score separate even if windows do not?
        fmean_v, fmean_d, flab = [], [], []
        for f in sorted(set(fl[test])):
            m = fl == f
            fmean_v.append(ev[m].mean()); fmean_d.append(dist[m].mean()); flab.append(int(y[m][0]))
        R['auc_vit_filemean'] = auc(flab, fmean_v); R['auc_meanspec_filemean'] = auc(flab, fmean_d)

        # --- 3. confounds ---------------------------------------------------
        if np.isfinite(temp).any():
            R['temp_mean_by_label'] = {int(l): float(np.nanmean(temp[y == l])) for l in np.unique(y)}
            ok = np.isfinite(temp) & (y == 0)
            R['corr_vit_temp_ref'] = float(np.corrcoef(ev[ok], temp[ok])[0, 1]) if ok.sum() > 10 else None
        R['corr_vit_power_ref'] = float(np.corrcoef(ev[y == 0], power[y == 0])[0, 1])
        R['corr_vit_pca_test'] = float(np.corrcoef(ev[test], ep[test])[0, 1])
        report[s] = R
        print('\n=== %s' % s)
        for k, v in R.items():
            print(' %-28s %s' % (k, json.dumps(v, default=str) if not isinstance(v, float) else '%.3f' % v))
    json.dump(report, open(os.path.join(DIAG, 'report.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
