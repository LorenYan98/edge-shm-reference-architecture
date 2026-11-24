"""Edge-latency probe for the PCA detector, the detector the manuscript ships.

Two steps:
  --fit   on the host: refit the PCA reference exactly as evaluate.py does (same
          training windows, same MinMax, 95 % variance) and save
          runs/<run>/pca.npz (mean, components, k, norm) per structure.
  (none)  inside the 4 vCPU / 4 GB container: for every raw window of
          cache/v2/raw_sample.npz, at batch size 1, time spectrogram ->
          normalize -> PCA residual -> threshold. Writes out/latency_pca.csv and
          out/latency_pca_info.json, same columns as latency.py.
"""
import argparse, csv, json, os, resource, time
import numpy as np
from . import frontend as fe_mod
from .datasets import cache_dir
from .spectrogram import MinMax

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRUCTURES = ['KW51', 'VANERSBORG', 'CHILLON']


def fit(run_of, frontend):
    import torch
    from .evaluate import PCARef
    from .train import RUNS, load_all, training_set
    for s, run in run_of.items():
        rd = os.path.join(RUNS, run)
        cfg = json.load(open(os.path.join(rd, 'config.json')))
        st = torch.load(os.path.join(rd, 'model.pt'), map_location='cpu')
        norm = MinMax(**st['norm'])
        data = load_all(cfg['structures'], seed=cfg['split_seed'], version=frontend)
        Xtr = norm(training_set(data, cfg['args']['per_structure'], cfg['split_seed']))
        pca = PCARef().fit(Xtr.reshape(len(Xtr), -1))
        np.savez(os.path.join(rd, 'pca.npz'), mean=pca.mean.astype(np.float32), comp=pca.comp.astype(np.float32),
                 k=pca.k, lo=norm.lo, hi=norm.hi, n_train=len(Xtr))
        print('%s: PCA-%d fitted on %d windows -> %s/pca.npz' % (s, pca.k, len(Xtr), rd), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='KW51=v2_KW51,VANERSBORG=v2_VANERSBORG,CHILLON=v2_CHILLON')
    ap.add_argument('--frontend', default='v2')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--hardware', default='container, 4 vCPU, 4 GB RAM, CPU only')
    ap.add_argument('--fit', action='store_true')
    a = ap.parse_args()
    run_of = dict(kv.split('=') for kv in a.runs.split(','))
    if a.fit:
        return fit(run_of, a.frontend)
    os.environ.setdefault('OMP_NUM_THREADS', str(a.threads))
    refs = {}
    for s, run in run_of.items():
        z = np.load(os.path.join(HERE, 'runs', run, 'pca.npz'))
        refs[s] = (z['mean'], z['comp'], MinMax(lo=float(z['lo']), hi=float(z['hi'])), fe_mod.get(a.frontend, s), int(z['k']))
    taus = {r['dataset']: float(r['threshold_value']) for r in csv.DictReader(open(os.path.join(HERE, 'out', 'runs.csv')))
            if r['model'].startswith('PCA')}

    def score(mean, comp, v):
        c = v - mean
        return float(((c - (c @ comp.T) @ comp) ** 2).mean())

    z = np.load(os.path.join(cache_dir(a.frontend), 'raw_sample.npz'), allow_pickle=True)
    X, ids, structs = z['x'], z['window_id'], z['structure']
    for s in refs:                                        # warm-up
        mean, comp, norm, fe, _ = refs[s]
        x0 = X[list(structs).index(s)]
        for _ in range(5):
            score(mean, comp, norm(fe.spectrogram(x0.astype(np.float64))).reshape(-1).astype(np.float32))
    rows = []
    for x, wid, s in zip(X, ids, structs):
        mean, comp, norm, fe, _ = refs[str(s)]
        t0 = time.perf_counter()
        sp = fe.spectrogram(x.astype(np.float64)); t1 = time.perf_counter()
        v = norm(sp).reshape(-1).astype(np.float32); t2 = time.perf_counter()
        e = score(mean, comp, v); t3 = time.perf_counter()
        pred = int(e > taus[str(s)]); t4 = time.perf_counter()
        rows.append([str(s), str(wid), '%.3f' % (1e3 * (t1 - t0)), '%.3f' % (1e3 * (t2 - t1)), '%.3f' % (1e3 * (t3 - t2)),
                     '%.3f' % (1e3 * (t4 - t3)), '%.3f' % (1e3 * (t4 - t0)), '%.6g' % e, pred])
    with open(os.path.join(HERE, 'out', 'latency_pca.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'window_id', 'ms_spectrogram', 'ms_normalize', 'ms_model', 'ms_threshold', 'latency_ms', 'pca_error', 'pca_pred'])
        w.writerows(rows)
    tot = np.array([float(r[6]) for r in rows])
    info = dict(hardware=a.hardware, threads=a.threads, n=len(rows), mean_ms=float(tot.mean()), p50=float(np.median(tot)),
                p95=float(np.quantile(tot, .95)), p99=float(np.quantile(tot, .99)), max=float(tot.max()),
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                pca_k={s: refs[s][4] for s in refs}, numpy=np.__version__)
    json.dump(info, open(os.path.join(HERE, 'out', 'latency_pca_info.json'), 'w'), indent=1); print(json.dumps(info))


if __name__ == '__main__':
    main()
