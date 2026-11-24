"""Re-score the trained v2 models with alternative anomaly scores, no retraining.

Scores per window (all from the saved model.pt of each structure):
  vit_unmasked   full-image reconstruction MSE, as in evaluate.py (reference)
  vit_masked     mean over K random 80 % masks of the masked-patch MSE (training regime)
  lat_maha       Mahalanobis distance of the unmasked encoder mean token in the
                 PCA-95 % subspace of the training latents
  lat_pca        reconstruction error of the encoder mean token outside that subspace
  latm_maha /    same two scores on the mean token averaged over the K masked passes
  latm_pca
  pca_pixels     the linear PCA baseline from evaluate.py (scores_<S>.npz)

Writes out/rescore/<structure>.npz (all scores + label + split) and
out/rescore/summary.json with AUC on the test split and sens/spec at
tau = median + 2.5 sd of the calibration split. Structures are processed in
the given order and the summary is rewritten after each one.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .evaluate import K_SD, confusion, rates, threshold
from .model import MAEViT
from .spectrogram import MinMax
from .train import RUNS, load_all, training_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'out', 'rescore')


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    r = s.argsort().argsort() + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def score_model(model, norm, specs, k_masks, batch, seed):
    n = len(specs)
    unm = np.empty(n); msk = np.zeros(n)
    lat = np.empty((n, 768), np.float32); latm = np.zeros((n, 768), np.float32)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for i in range(0, n, batch):
        x = torch.from_numpy(norm(specs[i:i + batch].astype(np.float32))).unsqueeze(1)
        target = model.patchify(x)
        z, _, _ = model.encode(x, masked=False)
        pred = model.decode(z, None)
        unm[i:i + batch] = ((pred - target) ** 2).mean(dim=(1, 2)).numpy()
        lat[i:i + batch] = z.mean(1).numpy()
        for _ in range(k_masks):
            torch.manual_seed(int(torch.randint(0, 2 ** 31, (1,), generator=g)))
            zv, mask, restore = model.encode(x, masked=True)
            pred = model.decode(zv, restore)
            per_patch = ((pred - target) ** 2).mean(-1)
            msk[i:i + batch] += ((per_patch * mask).sum(1) / mask.sum(1)).numpy() / k_masks
            latm[i:i + batch] += zv.mean(1).numpy() / k_masks
        if (i // batch) % 10 == 0:
            print('  %d/%d windows, %.0f s' % (min(i + batch, n), n, time.time() - t0), flush=True)
    return unm, msk, lat, latm


class LatentRef:
    """PCA-95 % subspace of training latents: Mahalanobis inside, residual outside."""

    def fit(self, Z, var=0.95):
        self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-6
        Zs = (Z - self.mu) / self.sd
        _, s, vt = np.linalg.svd(Zs, full_matrices=False)
        ev = s ** 2 / (s ** 2).sum()
        self.k = int(np.searchsorted(np.cumsum(ev), var) + 1)
        self.comp = vt[:self.k]; self.scale = s[:self.k] / np.sqrt(len(Z) - 1)
        return self

    def score(self, Z):
        Zs = (Z - self.mu) / self.sd
        proj = Zs @ self.comp.T
        maha = ((proj / self.scale) ** 2).sum(1)
        resid = ((Zs - proj @ self.comp) ** 2).mean(1)
        return maha, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='KW51=v2_KW51,VANERSBORG=v2_VANERSBORG,CHILLON=v2_CHILLON')
    ap.add_argument('--frontend', default='v2')
    ap.add_argument('--k-masks', type=int, default=4)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--threads', type=int, default=30)
    ap.add_argument('--max-train', type=int, default=2500, help='latent references fit on at most this many train windows')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    run_of = dict(kv.split('=') for kv in a.runs.split(','))
    sfile = os.path.join(OUT, 'summary.json')
    summary = json.load(open(sfile)) if os.path.exists(sfile) else {}
    for s, run in run_of.items():
        if s in summary and os.path.exists(os.path.join(OUT, '%s.npz' % s)):
            print('%s: already scored, skipping' % s, flush=True); continue
        rd = os.path.join(RUNS, run)
        cfg = json.load(open(os.path.join(rd, 'config.json')))
        st = torch.load(os.path.join(rd, 'model.pt'), map_location='cpu')
        model = MAEViT(img=st['img'], patch=st['patch']); model.load_state_dict(st['model']); model.eval()
        norm = MinMax(**st['norm'])
        (_, specs, metas, split), = load_all([s], seed=cfg['split_seed'], version=a.frontend)
        split = np.array(split); y = np.array([int(m['label']) for m in metas])
        train = np.where(split == 'train')[0]
        rng = np.random.default_rng(cfg['split_seed'])
        if len(train) > a.max_train:
            train = np.sort(rng.choice(train, a.max_train, replace=False))
        keep = np.sort(np.concatenate([train, np.where(split != 'train')[0]]))
        print('%s: scoring %d windows (%d train for refs, %d calib, %d test), K=%d masks' %
              (s, len(keep), len(train), (split == 'calib').sum(), (split == 'test').sum(), a.k_masks), flush=True)
        unm, msk, lat, latm = score_model(model, norm, specs[keep], a.k_masks, a.batch, cfg['split_seed'])
        sp, yy = split[keep], y[keep]
        tr = sp == 'train'; calib = sp == 'calib'; test = sp == 'test'
        ref = LatentRef().fit(lat[tr]); refm = LatentRef().fit(latm[tr])
        maha, lpca = ref.score(lat); maham, lpcam = refm.score(latm)
        scores = dict(vit_unmasked=unm, vit_masked=msk, lat_maha=maha, lat_pca=lpca, latm_maha=maham, latm_pca=lpcam)
        old = os.path.join(rd, 'scores_%s.npz' % s)
        if os.path.exists(old):
            z = np.load(old); scores['pca_pixels'] = z['pca'][keep]
            scores['vit_unmasked_prev'] = z['vit'][keep]
        np.savez(os.path.join(OUT, '%s.npz' % s), label=yy, split=sp, idx=keep, **scores)
        res = {}
        for name, e in scores.items():
            tau = threshold(e[calib])
            cm = confusion(yy[test], (e[test] > tau).astype(int))
            sens, spec, prec, f1 = rates(*cm)
            res[name] = dict(auc=auc(yy[test], e[test]) if yy[test].min() != yy[test].max() else None,
                             tau=tau, sens=sens, spec=spec, prec=prec, f1=f1, tp=cm[0], fn=cm[1], tn=cm[2], fp=cm[3])
            print('  %-18s AUC %s  sens %.3f spec %.3f' % (name, '%.3f' % res[name]['auc'] if res[name]['auc'] is not None else '  n/a', sens, spec), flush=True)
        res['_latent_k'] = dict(unmasked=ref.k, masked=refm.k)
        summary[s] = res
        json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'DONE'), 'w').close()


if __name__ == '__main__':
    main()
