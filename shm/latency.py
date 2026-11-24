"""Edge-latency probe. Run inside a resource-limited container
(docker run --cpus 4 --memory 4g) to emulate the 4 GB CPU-only edge node.

Reads <cache_dir>/raw_sample.npz (raw windows of test files), then for each
window at batch size 1 times: spectrogram -> normalize -> MAE-ViT score ->
threshold, with the structure's own front end and model. Writes
out/latency.csv (window_id, stage times, total) and out/latency_info.json.
"""
import argparse, csv, json, os, time, resource
import numpy as np, torch
from . import frontend as fe_mod
from .datasets import cache_dir
from .model import MAEViT
from .spectrogram import MinMax

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRUCTURES = ['KW51', 'VANERSBORG', 'CHILLON']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='mae_v1', help='one shared model run (v1)')
    ap.add_argument('--runs', default=None, help='per-structure runs, KW51=v2_KW51,...')
    ap.add_argument('--frontend', default='v1')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--hardware', default='container, 4 vCPU, 4 GB RAM, CPU only')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run_of = dict(kv.split('=') for kv in a.runs.split(',')) if a.runs else {s: a.run for s in STRUCTURES}
    models = {}
    for run in sorted(set(run_of.values())):
        st = torch.load(os.path.join(HERE, 'runs', run, 'model.pt'), map_location='cpu')
        model = MAEViT(img=st.get('img', (100, 100)), patch=st.get('patch', (5, 5)))
        model.load_state_dict(st['model']); model.eval()
        for s, r in run_of.items():
            if r == run:
                models[s] = (model, MinMax(**st['norm']), fe_mod.get(a.frontend, s))
    taus = {r['dataset']: float(r['threshold_value']) for r in csv.DictReader(open(os.path.join(HERE, 'out', 'runs.csv'))) if r['model'] == 'MAE-ViT'}
    z = np.load(os.path.join(cache_dir(a.frontend), 'raw_sample.npz'), allow_pickle=True)
    X, ids, structs = z['x'], z['window_id'], z['structure']
    for s in models:                                     # warm-up, every model
        model, norm, fe = models[s]
        x0 = X[list(structs).index(s)]
        for _ in range(5):
            model.score(torch.from_numpy(norm(fe.spectrogram(x0.astype(np.float64)))).reshape(1, 1, *fe.img))
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    rows = []
    for x, wid, s in zip(X, ids, structs):
        model, norm, fe = models[str(s)]
        t0 = time.perf_counter()
        sp = fe.spectrogram(x.astype(np.float64)); t1 = time.perf_counter()
        xt = torch.from_numpy(norm(sp)).reshape(1, 1, *fe.img); t2 = time.perf_counter()
        e = float(model.score(xt)[0]); t3 = time.perf_counter()
        pred = int(e > taus[str(s)]); t4 = time.perf_counter()
        rows.append([str(s), str(wid), '%.3f' % (1e3 * (t1 - t0)), '%.3f' % (1e3 * (t2 - t1)), '%.3f' % (1e3 * (t3 - t2)),
                     '%.3f' % (1e3 * (t4 - t3)), '%.3f' % (1e3 * (t4 - t0)), '%.6g' % e, pred])
    with open(os.path.join(HERE, 'out', 'latency.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['dataset', 'window_id', 'ms_spectrogram', 'ms_normalize', 'ms_model', 'ms_threshold', 'latency_ms', 'vit_error', 'vit_pred']); w.writerows(rows)
    tot = np.array([float(r[6]) for r in rows])
    info = dict(hardware=a.hardware, threads=a.threads, n=len(rows), mean_ms=float(tot.mean()), p50=float(np.median(tot)),
                p95=float(np.quantile(tot, .95)), p99=float(np.quantile(tot, .99)), max=float(tot.max()),
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, torch=torch.__version__)
    json.dump(info, open(os.path.join(HERE, 'out', 'latency_info.json'), 'w'), indent=1); print(json.dumps(info))


if __name__ == '__main__':
    main()
