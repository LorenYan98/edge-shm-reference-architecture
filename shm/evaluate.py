"""Score every cached window with the trained MAE-ViT (one shared model, or one
per structure with --runs) and a linear PCA reference fitted on the same
training windows, calibrate one threshold per structure on
the calibration split (median + 2.5 sd of normal-condition error), and write:

  out/<structure>_windows.csv   per-window export, manuscript schema (test split)
  out/runs.csv                  one row per (dataset, model): rule, tau, n, date
  out/<structure>_sweep.csv     metrics vs threshold multiplier k
  out/<structure>_roc.csv       ROC / PR points for ViT and PCA
  out/<structure>_files.csv     per-source-file mean error, time, label, temp
  out/train_curve.csv           copy of the training loss log
  out/summary.json
"""
import argparse
import csv
import datetime as dt
import json
import os
import shutil
import time

import numpy as np
import torch

from .model import MAEViT, n_params
from .spectrogram import MinMax
from .train import RUNS, STRUCTURES, load_all, training_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'out')
K_SD = 2.5


def vit_scores(model, norm, specs, batch=64):
    out = np.empty(len(specs), np.float64)
    with torch.no_grad():
        for i in range(0, len(specs), batch):
            x = torch.from_numpy(norm(specs[i:i + batch].astype(np.float32))).unsqueeze(1)
            out[i:i + batch] = model.score(x).numpy()
    return out


class PCARef:
    """Linear PCA reconstruction-error reference; k chosen to explain `var`
    of the training variance."""

    def __init__(self, var=0.95):
        self.var = var

    def fit(self, X):                       # X (n, 10000) normalized
        self.mean = X.mean(0)
        Xc = X - self.mean
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        ev = s ** 2 / (s ** 2).sum()
        self.k = int(np.searchsorted(np.cumsum(ev), self.var) + 1)
        self.comp = vt[:self.k]
        return self

    def score(self, X, batch=2048):
        out = np.empty(len(X))
        for i in range(0, len(X), batch):
            Xc = X[i:i + batch] - self.mean
            rec = (Xc @ self.comp.T) @ self.comp
            out[i:i + batch] = ((Xc - rec) ** 2).mean(1)
        return out


def threshold(calib_err, k=K_SD):
    return float(np.median(calib_err) + k * np.std(calib_err, ddof=1))


def confusion(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    return tp, fn, tn, fp


def rates(tp, fn, tn, fp):
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * prec * sens / (prec + sens) if (tp + fn) and (tp + fp) and prec + sens else float('nan')
    return sens, spec, prec, f1


def roc_points(y, e, n=200):
    qs = np.quantile(e, np.linspace(0, 1, n))
    rows = []
    for t in qs:
        tp, fn, tn, fp = confusion(y, (e > t).astype(int))
        sens, spec, prec, _ = rates(tp, fn, tn, fp)
        rows.append((t, 1 - spec, sens, prec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='mae_v1', help='one shared model run (v1)')
    ap.add_argument('--runs', default=None,
                    help='per-structure runs, e.g. KW51=v2_KW51,VANERSBORG=v2_VANERSBORG,CHILLON=v2_CHILLON')
    ap.add_argument('--frontend', default='v1')
    ap.add_argument('--threads', type=int, default=30)
    ap.add_argument('--hardware', default='development host, 32-core CPU')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    run_of = dict(kv.split('=') for kv in a.runs.split(',')) if a.runs else {s: a.run for s in STRUCTURES}

    def load_run(run):
        rd = os.path.join(RUNS, run)
        cfg = json.load(open(os.path.join(rd, 'config.json')))
        st = torch.load(os.path.join(rd, 'model.pt'), map_location='cpu')
        model = MAEViT(img=st.get('img', (100, 100)), patch=st.get('patch', (5, 5)))
        model.load_state_dict(st['model']); model.eval()
        return rd, cfg, model, MinMax(**st['norm'])

    # one PCA per model: fitted on exactly that model's training windows
    models, pcas, curves = {}, {}, []
    for run in sorted(set(run_of.values())):
        rd, cfg, model, norm = load_run(run)
        data_run = load_all(cfg['structures'], seed=cfg['split_seed'], version=a.frontend)
        Xtr = norm(training_set(data_run, cfg['args']['per_structure'], cfg['split_seed']))
        pca = PCARef().fit(Xtr.reshape(len(Xtr), -1))
        print('%s: PCA components for 95%% variance: %d (n_train %d)' % (run, pca.k, len(Xtr)))
        for s in cfg['structures']:
            models[s] = (rd, cfg, model, norm); pcas[s] = (pca, len(Xtr))
        with open(os.path.join(rd, 'train_log.csv')) as fh:
            for i, line in enumerate(fh):
                if i == 0 and curves:
                    continue
                curves.append((line.strip() + (',run' if i == 0 else ',' + run)))
    with open(os.path.join(OUT, 'train_curve.csv'), 'w') as fh:
        fh.write('\n'.join(curves) + '\n')

    data = load_all(seed=models[STRUCTURES[0]][1]['split_seed'], version=a.frontend)
    runs_rows, summary = [], {}
    today = dt.date.today().isoformat()
    for s, specs, metas, split in data:
        rd, cfg, model, norm = models[s]; pca, n_tr = pcas[s]
        t0 = time.time()
        sc_file = os.path.join(rd, 'scores_%s.npz' % s)
        if os.path.exists(sc_file):
            z = np.load(sc_file); ev, ep = z['vit'], z['pca']
        else:
            ev = vit_scores(model, norm, specs)
            ep = pca.score(norm(specs.astype(np.float32)).reshape(len(specs), -1))
            np.savez(sc_file, vit=ev, pca=ep)
        print('%s scored %d windows in %.0f s' % (s, len(specs), time.time() - t0))
        split = np.array(split); y = np.array([int(m['label']) for m in metas])
        calib = split == 'calib'; test = split == 'test'
        tau_v, tau_p = threshold(ev[calib]), threshold(ep[calib])
        pv, pp = (ev > tau_v).astype(int), (ep > tau_p).astype(int)

        with open(os.path.join(OUT, '%s_windows.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['dataset', 'window_id', 'period', 'label', 'vit_error', 'vit_pred', 'pca_error', 'pca_pred',
                        'latency_ms', 'placeholder', 'channel', 't0', 'temp', 'file'])
            for i in np.where(test)[0]:
                m = metas[i]
                w.writerow([s, '%s|%s|%s' % (m['file'], m['channel'], m['t0']), m['period'], y[i],
                            '%.6g' % ev[i], pv[i], '%.6g' % ep[i], pp[i], '', 0, m['channel'], m['t0'], m['temp'], m['file']])
        for name, tau in (('MAE-ViT', tau_v), ('PCA-%d' % pca.k, tau_p)):
            runs_rows.append([s, name, 'median + %.1f sd of calibration-split error' % K_SD, '%.6g' % tau,
                              a.hardware, int(test.sum()), today, 0])

        with open(os.path.join(OUT, '%s_sweep.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['k', 'tau', 'sens', 'spec', 'prec', 'f1', 'sens_pca', 'spec_pca', 'prec_pca'])
            for k in np.arange(0, 6.01, 0.25):
                r = rates(*confusion(y[test], (ev[test] > threshold(ev[calib], k)).astype(int)))
                q = rates(*confusion(y[test], (ep[test] > threshold(ep[calib], k)).astype(int)))
                w.writerow(['%.2f' % k, '%.6g' % threshold(ev[calib], k)] + ['%.4f' % v for v in r] + ['%.4f' % v for v in q[:3]])
        with open(os.path.join(OUT, '%s_roc.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['model', 'tau', 'fpr', 'tpr', 'prec'])
            for name, e in (('vit', ev[test]), ('pca', ep[test])):
                for row in roc_points(y[test], e):
                    w.writerow([name] + ['%.5g' % v for v in row])
        with open(os.path.join(OUT, '%s_files.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['file', 't0', 'label', 'split', 'temp', 'vit_mean', 'vit_max', 'pca_mean', 'n', 'frac_alarm'])
            files = sorted({m['file'] for m in metas})
            for f in files:
                idx = np.array([i for i, m in enumerate(metas) if m['file'] == f])
                m0 = metas[idx[0]]
                w.writerow([f, m0['t0'], y[idx[0]], split[idx[0]], m0['temp'], '%.6g' % ev[idx].mean(), '%.6g' % ev[idx].max(),
                            '%.6g' % ep[idx].mean(), len(idx), '%.4f' % pv[idx].mean()])

        cm_v = confusion(y[test], pv[test]); cm_p = confusion(y[test], pp[test])
        summary[s] = dict(n_test=int(test.sum()), n_calib=int(calib.sum()), n_train=int((split == 'train').sum()),
                          tau_vit=tau_v, tau_pca=tau_p, vit=dict(zip(['tp', 'fn', 'tn', 'fp'], cm_v)),
                          pca=dict(zip(['tp', 'fn', 'tn', 'fp'], cm_p)),
                          vit_rates=dict(zip(['sens', 'spec', 'prec', 'f1'], rates(*cm_v))),
                          pca_rates=dict(zip(['sens', 'spec', 'prec', 'f1'], rates(*cm_p))))
        print(s, json.dumps(summary[s]['vit_rates']), 'PCA', json.dumps(summary[s]['pca_rates']))

    with open(os.path.join(OUT, 'runs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'model', 'threshold_rule', 'threshold_value', 'hardware', 'n_windows', 'run_date', 'placeholder'])
        w.writerows(runs_rows)
    summary['_model'] = dict(params=n_params(model), pca_k={s: pcas[s][0].k for s in pcas},
                             n_train={s: pcas[s][1] for s in pcas}, runs=run_of, frontend=a.frontend,
                             img=list(model.img), patch=list(model.psz))
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
