"""Collect raw windows from TEST-split files (N per structure, both labels)
into <cache_dir>/raw_sample.npz for the latency probe. window_id matches the
per-window export so latencies can be joined back. Windows may differ in
length across structures (v2), so they are stored as an object array."""
import argparse, os
import numpy as np
from . import frontend as fe_mod
from .datasets import ITER, cache_dir, load_cache
from .train import STRUCTURES, split_files


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per-structure', type=int, default=400); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frontend', default='v1')
    a = ap.parse_args(); rng = np.random.RandomState(a.seed)
    xs, ids, ss = [], [], []
    for s in STRUCTURES:
        fe = fe_mod.get(a.frontend, s)
        specs, metas = load_cache(s, a.frontend); sp = split_files(metas, a.seed)
        test_files = sorted({m['file'] for m in metas if sp[m['split_key']] == 'test'})
        cached_ids = {'%s|%s|%s' % (m['file'], m['channel'], m['t0']) for m in metas if sp[m['split_key']] == 'test'}
        pick = set(rng.choice(test_files, min(len(test_files), 12), replace=False))
        got = []
        for meta, w in ITER[s](fe):
            if meta['file'] not in pick:
                continue
            wid = '%s|%s|%s' % (meta['file'], meta['channel'], meta['t0'])
            if wid in cached_ids:
                got.append((wid, w.astype(np.float32)))
        sel = rng.choice(len(got), min(a.per_structure, len(got)), replace=False)
        for i in sorted(sel):
            ids.append(got[i][0]); xs.append(got[i][1]); ss.append(s)
        print(s, len(sel), 'raw windows from', len(pick), 'test files')
    xo = np.empty(len(xs), dtype=object); xo[:] = xs
    np.savez_compressed(os.path.join(cache_dir(a.frontend), 'raw_sample.npz'), x=xo, window_id=np.array(ids), structure=np.array(ss))


if __name__ == '__main__':
    main()
