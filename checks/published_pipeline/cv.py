# usage: python3 -I cv.py <prep_npz> <model: pipe|raw|shape> <split: a|b|c1|c2|paper> <seed|gkf> <rounds> <lr> <out_npz> [cluster_npz] [unique]
import sys, time
import numpy as np
import lightgbm as lgb
from scipy.spatial import ConvexHull

PREP, MODEL, SPLIT, SEED, ROUNDS, LR, OUT = sys.argv[1:8]
CLU = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != '-' else None
UNIQUE = len(sys.argv) > 9 and sys.argv[9] == 'unique'
ROUNDS = int(ROUNDS); LR = float(LR)
p = np.load(PREP)
UX, UY, tp, cnt, inv, TP, fam = [p[k] for k in ('UX', 'UY', 'tp', 'cnt', 'inv', 'TP', 'fam')]
PWu = p['PWu']
nu, N = UX.shape
iu, ju = np.triu_indices(N, 1)

def theta_from_power():
    xc = UX - UX.mean(1, keepdims=True); yc = UY - UY.mean(1, keepdims=True); pc = PWu - PWu.mean(1, keepdims=True)
    A = np.stack([xc.ravel(), yc.ravel()], 1); b, *_ = np.linalg.lstsq(A, pc.ravel(), rcond=None)
    return np.arctan2(b[1], b[0])

def feats_pipe():
    dx = (UX[:, ju] - UX[:, iu]).astype(np.float32); dy = (UY[:, ju] - UY[:, iu]).astype(np.float32)
    d = np.hypot(dx, dy); ang = np.arctan2(dy, dx)
    c = dx / (d + 1e-8); s = dy / (d + 1e-8)
    st = np.stack([d.min(1), d.mean(1), np.median(d, 1), d.std(1)], 1)
    hull = np.array([ConvexHull(np.stack([UX[k], UY[k]], 1)).volume for k in range(nu)], np.float32)[:, None]
    base = np.concatenate([UX, UY, d, ang, st, hull], 1).astype(np.float32)
    return base, c, s

def feats_shape(theta):
    c, s = np.cos(theta), np.sin(theta)
    U = UX * c + UY * s; V = -UX * s + UY * c
    out = []
    for a in range(0, nu, 500):
        u = U[a:a + 500]; v = V[a:a + 500]; m = len(u)
        du = u[:, None, :] - u[:, :, None]; dv = v[:, None, :] - v[:, :, None]   # neighbour j minus buoy i
        d = np.hypot(du, dv); eye = np.eye(N, dtype=bool)[None]
        dd = np.where(eye, np.inf, d)
        nn = dd.min(2)
        pd_ = d[:, iu, ju]
        f = [nn.min(1), nn.mean(1), np.median(nn, 1), nn.std(1), nn.max(1),
             pd_.min(1), pd_.mean(1), np.median(pd_, 1), pd_.std(1), np.percentile(pd_, 10, 1), np.percentile(pd_, 90, 1),
             u.std(1), v.std(1), np.ptp(u, 1), np.ptp(v, 1)]
        for r in (50, 100, 150, 200):
            f.append((dd < r).sum(2).mean(1))
        ang = np.degrees(np.arctan2(dv, du)); up30 = (np.abs(ang) <= 30) & ~eye; up15 = (np.abs(ang) <= 15) & ~eye
        for r in (np.inf, 200, 400):
            k = up30 & (dd < r) if np.isfinite(r) else up30
            f.append(k.sum(2).mean(1)); f.append((~k.any(2)).mean(1))
        clr = np.where(up30, d, 1500.0).min(2)
        f += [clr.mean(1), np.median(clr, 1), clr.min(1), np.percentile(clr, 10, 1)]
        sh15 = np.where(up15, 1 / np.sqrt(np.maximum(d, 1)), 0).sum(2); sh30 = np.where(up30, 1 / np.sqrt(np.maximum(d, 1)), 0).sum(2)
        f += [sh15.mean(1), sh15.max(1), sh30.mean(1)]
        side = (np.abs(du) < 50) & ~eye
        sd_ = np.where(side, np.abs(dv), 1500.0).min(2)
        f += [sd_.mean(1), np.median(sd_, 1), (side & (dd < 150)).sum(2).mean(1)]
        uc = u - u.mean(1, keepdims=True); vc = v - v.mean(1, keepdims=True)
        f += [np.percentile(uc, 10, 1), np.percentile(uc, 90, 1), (uc ** 3).mean(1) / (u.std(1) ** 3 + 1e-9), (vc ** 3).mean(1) / (v.std(1) ** 3 + 1e-9),
              u.mean(1), v.mean(1)]
        cuu = (uc ** 2).mean(1); cvv = (vc ** 2).mean(1); cuv = (uc * vc).mean(1)
        tr = cuu + cvv; det = cuu * cvv - cuv ** 2; disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        phi = 0.5 * np.arctan2(2 * cuv, cuu - cvv)
        f += [tr / 2 + disc, tr / 2 - disc, np.cos(2 * phi), np.sin(2 * phi)]
        us = np.sort(u, 1); vs = np.sort(v, 1); gu = np.diff(us, axis=1); gv = np.diff(vs, axis=1)
        f += [(gu > 30).sum(1), gv.mean(1), gv.std(1), gu.std(1),
              (np.abs(du[:, iu, ju]) < 25).mean(1), (np.abs(dv[:, iu, ju]) < 25).mean(1), (u > u.max(1, keepdims=True) - 50).sum(1)]
        hull = np.array([ConvexHull(np.stack([u[k], v[k]], 1)).volume for k in range(m)])
        f.append(hull)
        out.append(np.stack(f, 1))
    return np.concatenate(out).astype(np.float32)

t0 = time.time()
if MODEL == 'pipe':
    base, Cc, Ss = feats_pipe()
elif MODEL == 'raw':
    base = np.concatenate([UX, UY], 1).astype(np.float32)
else:
    th = np.deg2rad(55.3) if 'perth' in PREP.lower() else theta_from_power()
    base = feats_shape(th)
    print('shape feats', base.shape[1], 'theta deg', round(float(np.degrees(th)), 1), 'theta_from_power', round(float(np.degrees(theta_from_power())), 1))
print('features', base.shape, 'built in', round(time.time() - t0, 1), 's', flush=True)

# ---- unit of analysis: rows (paper) or unique layouts
if UNIQUE:
    lay = np.arange(nu); y = tp.copy()
else:
    lay = inv; y = TP.copy()
n = len(lay)
if SPLIT in ('a', 'paper'):
    grp = np.arange(n)
elif SPLIT == 'b':
    grp = lay
elif SPLIT == 'c1':
    grp = fam[lay]
else:
    grp = np.load(CLU)['clu'][lay]

def assign(grp, seed):
    ug, gi = np.unique(grp, return_inverse=True)
    if seed == 'gkf':   # sklearn GroupKFold default (deterministic, biggest groups first)
        from sklearn.model_selection import GroupKFold
        fold = np.zeros(len(grp), int)
        for k, (_, te) in enumerate(GroupKFold(5).split(grp, groups=grp)):
            fold[te] = k
        return fold
    rng = np.random.default_rng(int(seed))
    if len(ug) == len(grp):
        perm = rng.permutation(len(grp)); fold = np.zeros(len(grp), int); fold[perm] = np.arange(len(grp)) % 5
        return fold
    # shuffled groups, then greedy balance on row counts (largest first, ties random)
    sz = np.bincount(gi); order = rng.permutation(len(ug)); order = order[np.argsort(-sz[order], kind='stable')]
    load = np.zeros(5); gf = np.zeros(len(ug), int)
    big = sz[order] > 1
    for g in order[big]:
        k = rng.choice(np.flatnonzero(load == load.min())); gf[g] = k; load[k] += sz[g]
    rest = order[~big]; rest = rng.permutation(rest)
    # fill singletons to equalise
    tgt = len(grp) / 5; pos = 0
    for k in np.argsort(load):
        need = int(round(tgt - load[k])) if k != np.argsort(load)[-1] else 0
        gf[rest[pos:pos + max(need, 0)]] = k; pos += max(need, 0)
    gf[rest[pos:]] = np.arange(len(rest) - pos) % 5
    return gf[gi]

if SPLIT == 'paper':
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(n), test_size=0.20, random_state=42)
    fold = np.ones(n, int); fold[te] = 0; folds = [0]
    print('paper split train', len(tr), 'test', len(te))
else:
    fold = assign(grp, SEED); folds = range(5)
mseed = 42 if SEED == 'gkf' or SPLIT == 'paper' else 42 + int(SEED)
params = dict(objective='regression', learning_rate=LR, num_leaves=64, bagging_fraction=0.8, feature_fraction=0.8,
              seed=mseed, num_threads=1, verbose=-1, force_col_wise=True)
pred = np.full(n, np.nan)
for k in folds:
    t1 = time.time()
    trm = fold != k; tem = ~trm
    Xtr = base[lay[trm]]
    ute = np.unique(lay[tem]); Xte = base[ute]
    if MODEL == 'pipe':
        w = np.bincount(lay[trm], minlength=nu).astype(float)
        mx = (w[:, None] * UX).sum() / (w.sum() * N); my = (w[:, None] * UY).sum() / (w.sum() * N)
        sxx = (w[:, None] * (UX - mx) ** 2).sum(); syy = (w[:, None] * (UY - my) ** 2).sum(); sxy = (w[:, None] * (UX - mx) * (UY - my)).sum()
        ev, evec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]])); vd = evec[:, -1]
        al = (Cc * vd[0] + Ss * vd[1]).astype(np.float32)
        Xtr = np.concatenate([Xtr, al[lay[trm]]], 1); Xte = np.concatenate([Xte, al[ute]], 1); del al
    ds = lgb.Dataset(Xtr, y[trm], free_raw_data=True); ds.construct(); nf = Xtr.shape[1]; del Xtr
    bst = lgb.train(params, ds, num_boost_round=ROUNDS)
    pu = np.full(nu, np.nan); pu[ute] = bst.predict(Xte)
    pred[tem] = pu[lay[tem]]
    r2 = 1 - ((y[tem] - pred[tem]) ** 2).sum() / ((y[tem] - y[tem].mean()) ** 2).sum()
    print(f'fold {k} n_test {tem.sum()} nfeat {nf} R2 {r2:.4f} rmse_kW {np.sqrt(((y[tem]-pred[tem])**2).mean())/1e3:.2f} time {time.time()-t1:.0f}s', flush=True)
    del ds, bst
m = ~np.isnan(pred)
r2 = 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
print(f'POOLED {MODEL} {SPLIT} seed {SEED} rounds {ROUNDS} lr {LR} unique {UNIQUE} R2 {r2:.4f}', flush=True)
np.savez_compressed(OUT, pred=pred, fold=fold, y=y, lay=lay)
