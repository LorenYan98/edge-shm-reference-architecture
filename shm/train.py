"""Split cached windows by source file and train a MAE-ViT on normal-condition
windows (self-supervised, 80 % masking): one model over all structures (v1) or
one model per structure (--structures KW51, v2).

Split rule (per structure, label-0 files only, seeded): 60 % of files train,
20 % calibration (threshold), 20 % test. Label-1 files are test only. Whole
files stay together so no window of a test file leaks into training.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from . import frontend as fe_mod
from .datasets import cache_dir, load_cache
from .model import MAEViT, n_params
from .spectrogram import MinMax

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, 'runs')
STRUCTURES = ['KW51', 'VANERSBORG', 'CHILLON']


def split_files(metas, seed=0):
    """Return dict split_key -> 'train'|'calib'|'test'."""
    rng = np.random.RandomState(seed)
    keys0 = sorted({m['split_key'] for m in metas if m['label'] == '0'})
    keys1 = sorted({m['split_key'] for m in metas if m['label'] == '1'})
    rng.shuffle(keys0)
    n = len(keys0); a, b = int(0.6 * n), int(0.8 * n)
    out = {k: 'train' for k in keys0[:a]}
    out.update({k: 'calib' for k in keys0[a:b]})
    out.update({k: 'test' for k in keys0[b:]})
    out.update({k: 'test' for k in keys1})
    return out


def load_all(structures=STRUCTURES, seed=0, version='v1'):
    """Returns list of (structure, specs, metas, split-per-row)."""
    out = []
    for s in structures:
        if not os.path.exists(os.path.join(cache_dir(version), s + '.npz')):
            continue
        specs, metas = load_cache(s, version)
        sp = split_files(metas, seed)
        rows = [sp[m['split_key']] for m in metas]
        out.append((s, specs, metas, rows))
    return out


def training_set(data, per_structure, seed=0):
    rng = np.random.RandomState(seed)
    parts = []
    for s, specs, metas, rows in data:
        idx = np.array([i for i, r in enumerate(rows) if r == 'train'])
        if len(idx) > per_structure:
            idx = rng.choice(idx, per_structure, replace=False)
        parts.append(specs[np.sort(idx)].astype(np.float32))
        print('  %s: %d training windows' % (s, len(idx)))
    return np.concatenate(parts)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='mae_v1')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--per-structure', type=int, default=2500)
    ap.add_argument('--threads', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frontend', default='v1', help='front-end version (cache/<version>, model geometry)')
    ap.add_argument('--structures', default=','.join(STRUCTURES), help='comma list; one model is trained on their union')
    ap.add_argument('--min-steps', type=int, default=0,
                    help='raise --epochs so that at least this many optimizer steps are taken')
    a = ap.parse_args()

    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rd = os.path.join(RUNS, a.run); os.makedirs(rd, exist_ok=True)
    structures = a.structures.split(',')
    data = load_all(structures, seed=a.seed, version=a.frontend)
    print('structures:', [d[0] for d in data])
    X = training_set(data, a.per_structure, a.seed)
    norm = MinMax().fit(X)
    X = torch.from_numpy(norm(X)).unsqueeze(1)
    print('training tensor', tuple(X.shape), 'min/max', float(X.min()), float(X.max()))

    fe = fe_mod.get(a.frontend, structures[0])
    assert all(fe_mod.get(a.frontend, s).img == fe.img for s in structures), 'mixed image sizes in one model'
    model = MAEViT(img=fe.img, patch=fe.patch)
    print('model params %.1f M, image %s, patch %s, tokens %d' % (n_params(model) / 1e6, fe.img, fe.patch, model.npatch))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    steps_per_epoch = math.ceil(len(X) / a.batch)
    if a.min_steps:
        a.epochs = max(a.epochs, math.ceil(a.min_steps / steps_per_epoch))
    total = steps_per_epoch * a.epochs; warm = steps_per_epoch * 2
    print('epochs %d x %d steps' % (a.epochs, steps_per_epoch))
    start_epoch, step = 0, 0
    ck = os.path.join(rd, 'last.pt')
    if os.path.exists(ck):
        st = torch.load(ck, map_location='cpu')
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        start_epoch, step = st['epoch'] + 1, st['step']
        print('resumed at epoch', start_epoch)
    json.dump(dict(norm=norm.state(), args=vars(a), n_train=len(X), params=n_params(model),
                   split_seed=a.seed, structures=[d[0] for d in data], frontend=a.frontend,
                   img=list(fe.img), patch=list(fe.patch), epochs=a.epochs, steps_per_epoch=steps_per_epoch),
              open(os.path.join(rd, 'config.json'), 'w'), indent=1)

    logf = open(os.path.join(rd, 'train_log.csv'), 'a', newline='')
    log = csv.writer(logf)
    if start_epoch == 0:
        log.writerow(['epoch', 'loss', 'lr', 'seconds'])
    g = torch.Generator().manual_seed(a.seed)
    for epoch in range(start_epoch, a.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        perm = torch.randperm(len(X), generator=g)
        for b in range(steps_per_epoch):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            lr = lr_at(step, total, warm, a.lr)
            for pg in opt.param_groups:
                pg['lr'] = lr
            loss, _, _ = model(X[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            tot += float(loss); nb += 1
            if b % 20 == 0:
                print('epoch %d step %d/%d loss %.5f lr %.2e  %.0fs' % (epoch, b, steps_per_epoch, float(loss), lr, time.time() - t0), flush=True)
        log.writerow([epoch, '%.6f' % (tot / nb), '%.3e' % lr, '%.0f' % (time.time() - t0)]); logf.flush()
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step), ck)
        print('== epoch %d mean loss %.5f in %.0f s' % (epoch, tot / nb, time.time() - t0), flush=True)
    torch.save(dict(model=model.state_dict(), norm=norm.state(), img=list(fe.img), patch=list(fe.patch),
                    frontend=a.frontend, structures=[d[0] for d in data]), os.path.join(rd, 'model.pt'))
    print('saved', os.path.join(rd, 'model.pt'))


if __name__ == '__main__':
    main()
