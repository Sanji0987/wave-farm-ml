#!/usr/bin/env python3
"""Chain-pair emulation, N, O. Usage: python3 -I s05_rest.py <npz dir>"""
import sys, os
import numpy as np
d = sys.argv[1]
def load(k):
    z = np.load(os.path.join(d, k + ".npz")); return z["X"], z["Y"], z["P"], z["q"], z["T"], int(z["scale"])
X, Y, P, q, T, sc = load("Perth_49")
XY = np.concatenate([X, Y], axis=1)
Uq, inv = np.unique(XY, axis=0, return_inverse=True); inv = inv.ravel(); U = len(Uq); N = 49
Tm = np.bincount(inv, weights=T) / np.bincount(inv)
UX = Uq[:, :N]; UY = Uq[:, N:]
pos = {}; L = np.empty((U, N), dtype=np.int32)
for i in range(U):
    for j in range(N):
        k = (int(UX[i, j]), int(UY[i, j])); v = pos.get(k)
        if v is None: v = len(pos); pos[k] = v
        L[i, j] = v
pxy = np.array(list(pos.keys())) / sc
buckets = {}
for i in range(U):
    o = np.argsort(L[i]); row = L[i][o]
    for j in range(N):
        buckets.setdefault(np.delete(row, j).tobytes(), []).append((i, int(row[j])))
sz = np.array([len(v) for v in buckets.values()])
chain = []; allp = 0
for v in buckets.values():
    allp += len(v) * (len(v) - 1) // 2
    for a, b in zip(v[:-1], v[1:]):
        chain.append(np.hypot(*(pxy[a[1]] - pxy[b[1]])))
print(f"buckets with >=2 layouts: {int((sz>1).sum())}; sizes: 2:{int((sz==2).sum())} 3-10:{int(((sz>2)&(sz<=10)).sum())} >10:{int((sz>10).sum())} max={sz.max()}")
print(f"all pairs={allp}; adjacent-only (chain) pairs={len(chain)}; chain median move={np.median(chain):.1f} m")
# team's fold seed with full pairs vs chain pairs, in np.unique order
rng = np.random.default_rng(0); fold = rng.integers(0, 5, U)
lk_c = np.zeros(U, bool); lk_a = np.zeros(U, bool)
for v in buckets.values():
    if len(v) < 2: continue
    ids = np.array([a[0] for a in v]); f = fold[ids]
    x = f[:-1] != f[1:]; lk_c[ids[:-1][x]] = True; lk_c[ids[1:][x]] = True
    lk_a[ids[np.array([(f != fi).any() for fi in f])]] = True
print(f"leak share seed 0 (team fold draw): chain pairs {lk_c.mean()*100:.2f}% ; all pairs {lk_a.mean()*100:.2f}%")

# N: orientation of top-5% layouts
top = np.argsort(-Tm)[: int(round(0.05 * U))]
for kk in (1, 2, 4):
    ang = []
    for i in top:
        x = UX[i] / sc; y = UY[i] / sc
        dx = x[None, :] - x[:, None]; dy = y[None, :] - y[:, None]; D = np.hypot(dx, dy); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :kk]; r = np.arange(N)[:, None]
        ang.append((np.degrees(np.arctan2(dy[r, nn], dx[r, nn])) % 180).ravel())
    ang = np.concatenate(ang); h = np.bincount(np.floor(ang).astype(int) % 180, minlength=180)
    o = np.argsort(-h)[:5]
    # axial mean direction
    z = np.exp(2j * np.deg2rad(ang)).mean()
    print(f"N: top-5% (n={len(top)}), {kk} nearest neighbours: top 1-deg bins (deg:share%) {[(int(a), round(h[a]/h.sum()*100,1)) for a in o]}; share within 50-62: {h[50:62].sum()/h.sum()*100:.1f}%; 140-152: {h[140:152].sum()/h.sum()*100:.1f}%; axial mean {np.degrees(np.angle(z))/2%180:.1f} (R={abs(z):.2f})")
best = int(np.argmax(Tm)); x = UX[best] / sc; y = UY[best] / sc
dx = x[None, :] - x[:, None]; dy = y[None, :] - y[:, None]; D = np.hypot(dx, dy); np.fill_diagonal(D, np.inf)
j = D.argmin(1); a = np.degrees(np.arctan2(dy[np.arange(N), j], dx[np.arange(N), j])) % 180
print("   best layout nearest-neighbour bearings (deg mod 180) histogram 10-deg bins:", np.bincount((a // 10).astype(int), minlength=18).tolist(), "NN dist median", round(float(np.median(D.min(1))), 1))

# O: Sydney 49
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
X, Y, P, q, T, sc = load("Sydney_49")
XY = np.concatenate([X, Y], axis=1); Uq, inv = np.unique(XY, axis=0, return_inverse=True); inv = inv.ravel()
Tm = np.bincount(inv, weights=T) / np.bincount(inv)
x = Uq[:, :49] / sc; y = Uq[:, 49:] / sc; iu = np.triu_indices(49, 1)
mpd = np.array([np.hypot(x[i][:, None] - x[i][None, :], y[i][:, None] - y[i][None, :])[iu].mean() for i in range(len(Uq))])
F = mpd[:, None]
print(f"O: Sydney49 unique={len(Uq)}; Pearson r={np.corrcoef(mpd, Tm)[0,1]:.4f}; Spearman={np.corrcoef(np.argsort(np.argsort(mpd)), np.argsort(np.argsort(Tm)))[0,1]:.4f}")
for nm, mk in [("linear", lambda: LinearRegression()), ("GBM(200,d3,lr0.05)", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0))]:
    ins = r2_score(Tm, mk().fit(F, Tm).predict(F))
    cv = [r2_score(Tm, cross_val_predict(mk(), F, Tm, cv=KFold(5, shuffle=True, random_state=s))) for s in (0, 1, 2)]
    print(f"   {nm}: in-sample R2={ins:.4f}; 5-fold CV (pooled) R2={np.mean(cv):.4f} (seeds 0-2: {np.round(cv,4)})")
# all rows (no dedup), linear + GBM row-split CV
Fr = mpd[inv][:, None]
print(f"   all rows (dups kept): linear in-sample R2={r2_score(T, LinearRegression().fit(Fr, T).predict(Fr)):.4f}; GBM row-wise 5-fold CV R2={r2_score(T, cross_val_predict(GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0), Fr, T, cv=KFold(5, shuffle=True, random_state=0))):.4f}")
