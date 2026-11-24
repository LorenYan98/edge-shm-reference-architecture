"""Fold the masked-scoring ViT results (out/rescore/<S>.npz) into the paper export in out/.

The manuscript reports the PCA detector as deployed and the MAE-ViT as the learned
alternative scored in its training regime (masked-patch MSE averaged over K masks).
evaluate.py wrote the ViT columns with the unmasked full-image score; this script
replaces them with the masked score, keeps the unmasked score in extra columns, and
recomputes everything derived from the ViT score (threshold, predictions, sweep, ROC,
per-file means, confusion matrices). PCA columns are rewritten from the same npz and
are identical to evaluate.py's. Structures without an npz are left untouched.
"""
import csv
import json
import os

import numpy as np

from .evaluate import K_SD, OUT, STRUCTURES, confusion, rates, roc_points, threshold
from .rescore import OUT as RESCORE, auc
from .train import RUNS, load_all


def main():
    summary = json.load(open(os.path.join(OUT, 'summary.json')))
    run_of = summary['_model']['runs']
    runs_rows = list(csv.DictReader(open(os.path.join(OUT, 'runs.csv'))))
    done = []
    for s in STRUCTURES:
        npz = os.path.join(RESCORE, '%s.npz' % s)
        if not os.path.exists(npz):
            print('%s: no rescore yet, left as is' % s); continue
        z = np.load(npz, allow_pickle=True)
        cfg = json.load(open(os.path.join(RUNS, run_of[s], 'config.json')))
        (_, _, metas, split), = load_all([s], seed=cfg['split_seed'], version=summary['_model']['frontend'])
        split = np.array(split); y = np.array([int(m['label']) for m in metas])
        keep = z['idx']
        assert (np.array(z['split']) == split[keep]).all() and (z['label'] == y[keep]).all(), s
        n = len(metas)
        ev = np.full(n, np.nan); eu = np.full(n, np.nan); ep = np.full(n, np.nan)
        ev[keep] = z['vit_masked']; eu[keep] = z['vit_unmasked']; ep[keep] = z['pca_pixels']
        calib = split == 'calib'; test = split == 'test'
        tau_v, tau_u, tau_p = threshold(ev[calib]), threshold(eu[calib]), threshold(ep[calib])
        pv, pu, pp = (ev > tau_v).astype(int), (eu > tau_u).astype(int), (ep > tau_p).astype(int)

        with open(os.path.join(OUT, '%s_windows.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['dataset', 'window_id', 'period', 'label', 'vit_error', 'vit_pred', 'pca_error', 'pca_pred',
                        'latency_ms', 'placeholder', 'channel', 't0', 'temp', 'file', 'vit_unmasked_error', 'vit_unmasked_pred'])
            for i in np.where(test)[0]:
                m = metas[i]
                w.writerow([s, '%s|%s|%s' % (m['file'], m['channel'], m['t0']), m['period'], y[i],
                            '%.6g' % ev[i], pv[i], '%.6g' % ep[i], pp[i], '', 0, m['channel'], m['t0'], m['temp'], m['file'],
                            '%.6g' % eu[i], pu[i]])
        for r in runs_rows:
            if r['dataset'] == s and r['model'] == 'MAE-ViT':
                r['threshold_value'] = '%.6g' % tau_v
        with open(os.path.join(OUT, '%s_sweep.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['k', 'tau', 'sens', 'spec', 'prec', 'f1', 'sens_pca', 'spec_pca', 'prec_pca'])
            for k in np.arange(0, 6.01, 0.25):
                r = rates(*confusion(y[test], (ev[test] > threshold(ev[calib], k)).astype(int)))
                q = rates(*confusion(y[test], (ep[test] > threshold(ep[calib], k)).astype(int)))
                w.writerow(['%.2f' % k, '%.6g' % threshold(ev[calib], k)] + ['%.4f' % v for v in r] + ['%.4f' % v for v in q[:3]])
        with open(os.path.join(OUT, '%s_roc.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['model', 'tau', 'fpr', 'tpr', 'prec'])
            for name, e in (('vit', ev[test]), ('pca', ep[test]), ('vit_unmasked', eu[test])):
                for row in roc_points(y[test], e):
                    w.writerow([name] + ['%.5g' % v for v in row])
        with open(os.path.join(OUT, '%s_files.csv' % s.lower()), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 't0', 'label', 'split', 'temp', 'vit_mean', 'vit_max', 'pca_mean', 'n', 'frac_alarm', 'pca_max', 'frac_alarm_pca'])
            by_file = {}
            for i in keep:
                by_file.setdefault(metas[i]['file'], []).append(i)
            for f in sorted(by_file):
                idx = np.array(by_file[f]); m0 = metas[idx[0]]
                w.writerow([f, m0['t0'], y[idx[0]], split[idx[0]], m0['temp'], '%.6g' % ev[idx].mean(), '%.6g' % ev[idx].max(),
                            '%.6g' % ep[idx].mean(), len(idx), '%.4f' % pv[idx].mean(), '%.6g' % ep[idx].max(), '%.4f' % pp[idx].mean()])

        cm_v, cm_u, cm_p = confusion(y[test], pv[test]), confusion(y[test], pu[test]), confusion(y[test], pp[test])
        has_pos = y[test].min() != y[test].max()
        summary[s].update(dict(
            tau_vit=tau_v, tau_vit_unmasked=tau_u, tau_pca=tau_p,
            vit=dict(zip(['tp', 'fn', 'tn', 'fp'], cm_v)), vit_unmasked=dict(zip(['tp', 'fn', 'tn', 'fp'], cm_u)),
            pca=dict(zip(['tp', 'fn', 'tn', 'fp'], cm_p)),
            vit_rates=dict(zip(['sens', 'spec', 'prec', 'f1'], rates(*cm_v))),
            vit_unmasked_rates=dict(zip(['sens', 'spec', 'prec', 'f1'], rates(*cm_u))),
            pca_rates=dict(zip(['sens', 'spec', 'prec', 'f1'], rates(*cm_p))),
            auc=dict(vit=auc(y[test], ev[test]) if has_pos else None, vit_unmasked=auc(y[test], eu[test]) if has_pos else None,
                     pca=auc(y[test], ep[test]) if has_pos else None)))
        done.append(s)
        print('%s: PCA AUC %s sens %.3f spec %.3f | ViT masked AUC %s sens %.3f spec %.3f | ViT unmasked AUC %s' % (
            s, '%.3f' % summary[s]['auc']['pca'] if has_pos else 'n/a', *rates(*cm_p)[:2],
            '%.3f' % summary[s]['auc']['vit'] if has_pos else 'n/a', *rates(*cm_v)[:2],
            '%.3f' % summary[s]['auc']['vit_unmasked'] if has_pos else 'n/a'))

    with open(os.path.join(OUT, 'runs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs_rows[0].keys())); w.writeheader(); w.writerows(runs_rows)
    summary['_model']['vit_score'] = 'masked-patch MSE, mean over 4 random 80 % masks'
    summary['_model']['merged'] = done
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print('merged:', done)


if __name__ == '__main__':
    main()
