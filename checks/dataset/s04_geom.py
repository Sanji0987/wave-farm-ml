#!/usr/bin/env python3
"""Claims H-M, P (+ Perth-100 row 850). Usage: python3 -I s04_geom.py <npz dir>"""
import sys, os
import numpy as np
import scipy.sparse as sp
d = sys.argv[1]
np.set_printoptions(linewidth=200, suppress=True, precision=2)
def load(k):
    z = np.load(os.path.join(d, k + ".npz")); return z["X"], z["Y"], z["P"], z["q"], z["T"], int(z["scale"])
def groups(X, Y):
    seen = {}; gid = np.empty(len(X), dtype=np.int64); first = []
    for i in range(len(X)):
        key = (X[i].tobytes(), Y[i].tobytes()); g = seen.get(key)
        if g is None: g = len(first); seen[key] = g; first.append(i)
        gid[i] = g
    return gid, np.array(first)

X, Y, P, q, T, sc = load("Perth_49")
gid, first = groups(X, Y); U = len(first); N = 49
cnt = np.bincount(gid)
Pm = np.zeros((U, N)); np.add.at(Pm, gid, P); Pm /= cnt[:, None]
Tm = np.bincount(gid, weights=T) / cnt
Tmax = np.full(U, -np.inf); Tmin = np.full(U, np.inf); np.maximum.at(Tmax, gid, T); np.minimum.at(Tmin, gid, T)
clean = (Tmax - Tmin) == 0
Xu = X[first] / sc; Yu = Y[first] / sc
Pf = P[first]

def meancorr(theta, PP=Pm, mask=None):
    t = np.deg2rad(theta); u = Xu * np.cos(t) + Yu * np.sin(t)
    a = u - u.mean(1, keepdims=True); b = PP - PP.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return r.mean() if mask is None else r[mask].mean()
print(f"H: mean corr at 55.3: avg-dups {meancorr(55.3):.4f}; first-occ {meancorr(55.3, Pf):.4f}; excluding 235 groups {meancorr(55.3, Pm, clean):.4f}")
th = np.arange(0, 360, 1.0); mc = np.array([meancorr(a) for a in th])
b = th[np.argmax(mc)]; th2 = np.arange(b - 2, b + 2.01, 0.1); mc2 = np.array([meancorr(a) for a in th2])
print(f"   scan: best theta={th2[np.argmax(mc2)]:.1f} corr={mc2.max():.4f}; min at {th[np.argmin(mc)]:.0f} ({mc.min():.4f}); corr at 45/50/52/55/56/59/60/65: {[round(meancorr(a),4) for a in (45,50,52,55,56,59,60,65)]}")
ok = th2[mc2 >= mc2.max() - 0.005]; print(f"   theta range within 0.005 of max: {ok.min():.1f}..{ok.max():.1f}")

t = np.deg2rad(55.3); c, s = np.cos(t), np.sin(t)
u = Xu * c + Yu * s
rank = np.argsort(np.argsort(-u, axis=1, kind="stable"), axis=1, kind="stable")
def thirds(PP, mask=slice(None)):
    r = rank[mask]; p = PP[mask]
    f = p[r < 49 / 3].mean(); m = p[(r >= 49 / 3) & (r < 2 * 49 / 3)].mean(); bk = p[r >= 2 * 49 / 3].mean()
    return f / 1e3, m / 1e3, bk / 1e3, (bk / f - 1) * 100
print("I: thirds avg-dups  F/M/B kW, back-vs-front %%: %.2f %.2f %.2f %.2f" % thirds(Pm))
print("   first-occ: %.2f %.2f %.2f %.2f ; excl 235 groups: %.2f %.2f %.2f %.2f" % (thirds(Pf) + thirds(Pm, clean)))
print("   counts per third:", int((rank[0] < 49/3).sum()), int(((rank[0] >= 49/3) & (rank[0] < 98/3)).sum()), int((rank[0] >= 98/3).sum()), " all-rows (no dedup): %.2f %.2f %.2f %.2f" % (lambda r, p: (p[r < 49/3].mean()/1e3, p[(r >= 49/3) & (r < 98/3)].mean()/1e3, p[r >= 98/3].mean()/1e3, (p[r >= 98/3].mean()/p[r < 49/3].mean()-1)*100))(rank[gid], P))

# J
has = np.zeros((U, N), dtype=bool); cos30 = np.cos(np.deg2rad(30))
for a in range(0, U, 500):
    dx = Xu[a:a+500, None, :] - Xu[a:a+500, :, None]; dy = Yu[a:a+500, None, :] - Yu[a:a+500, :, None]  # vector i->j
    dist = np.sqrt(dx * dx + dy * dy); along = dx * c + dy * s
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (along / dist >= cos30 - 1e-12) & (dist > 0)
    has[a:a+500] = m.any(2)
for nm, PP, mk in [("avg-dups", Pm, slice(None)), ("first-occ", Pf, slice(None)), ("excl235", Pm, clean)]:
    h = has[mk]; p = PP[mk]
    print(f"J[{nm}]: no up-wave neighbour {p[~h].mean()/1e3:.2f} kW (n={int((~h).sum())}, {np.mean(~h)*100:.1f}% of buoys) vs rest {p[h].mean()/1e3:.2f} kW")

# K twins, exact
def twins(Xi, Yi, label):
    pos = {}; L = np.empty((U, N), dtype=np.int32)
    for i in range(U):
        for j in range(N):
            k = (int(Xi[i, j]), int(Yi[i, j])); v = pos.get(k)
            if v is None: v = len(pos); pos[k] = v
            L[i, j] = v
    L.sort(axis=1)
    dupin = int((np.diff(L, axis=1) == 0).any(1).sum())
    nset = len({r.tobytes() for r in L})
    keys = {}
    for i in range(U):
        row = L[i]
        for j in range(N):
            keys.setdefault(np.delete(row, j).tobytes(), []).append(i)
    pairs = set()
    for v in keys.values():
        if len(v) > 1:
            v = sorted(set(v))
            for a in range(len(v)):
                for b2 in range(a + 1, len(v)):
                    if L[v[a]].tobytes() != L[v[b2]].tobytes(): pairs.add((v[a], v[b2]))
    # independent check: incidence matrix product
    M = sp.csr_matrix((np.ones(U * N, dtype=np.int32), (np.repeat(np.arange(U), N), L.ravel())), shape=(U, len(pos)))
    M.data[:] = 1; n48 = 0; n49 = 0; hist = np.zeros(50, dtype=np.int64)
    for a in range(0, U, 1000):
        C = (M[a:a+1000] @ M.T).toarray()
        ii, jj = np.nonzero(C >= 40)
        keep = jj > (ii + a)
        hist += np.bincount(C[ii[keep], jj[keep]], minlength=50)[:50]
    print(f"K[{label}]: distinct positions={len(pos)}; layouts with repeated position={dupin}; distinct sets={nset}; twin pairs (minus-one keys)={len(pairs)}; incidence check shared==48: {hist[48]}, ==49: {hist[49]}, ==47: {hist[47]}, 40-46: {hist[40:47].sum()}")
    return np.array(sorted(pairs)), L
pairs, L = twins(X[first], Y[first], "exact 0.01 m")
pr, _ = twins(np.floor(X[first] / 10 + 0.5).astype(np.int64), np.floor(Y[first] / 10 + 0.5).astype(np.int64), "rounded 0.1 m")
# moved distance
mv = np.empty(len(pairs))
for k, (a, b2) in enumerate(pairs):
    sa = set(zip(X[first[a]].tolist(), Y[first[a]].tolist())); sb = set(zip(X[first[b2]].tolist(), Y[first[b2]].tolist()))
    (pa,) = sa - sb; (pb,) = sb - sa
    mv[k] = np.hypot(pa[0] - pb[0], pa[1] - pb[1]) / sc
print(f"   moved distance: median={np.median(mv):.2f} m mean={mv.mean():.1f} min={mv.min():.2f} max={mv.max():.1f}; quartiles {np.percentile(mv,[25,75])}")
par = np.arange(U)
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for a, b2 in pairs:
    ra, rb = find(a), find(b2)
    if ra != rb: par[ra] = rb
root = np.array([find(i) for i in range(U)]); _, fam, fs = np.unique(root, return_inverse=True, return_counts=True)
deg = np.bincount(pairs.ravel(), minlength=U)
ib = int(np.argmax(Tm)); fb = fam[ib]
print(f"   families={len(fs)} singletons={int((fs==1).sum())} largest={fs.max()} ({fs.max()/U*100:.2f}%) second={np.sort(fs)[-2]}; best layout in largest: {fs[fb]==fs.max()}; layouts with >=1 twin: {np.mean(deg>0)*100:.2f}% ({int((deg>0).sum())}); layouts in non-singleton families {int(fs[fs>1].sum())}")
inb = fam == fb
print(f"P: best family size={int(inb.sum())}; Total_Power kW min/median/max = {Tm[inb].min()/1e3:.1f} / {np.median(Tm[inb])/1e3:.1f} / {Tm[inb].max()/1e3:.1f}; rows in it={int(cnt[inb].sum())}; of the top-557 layouts by power, in family: {int(inb[np.argsort(-Tm)[:557]].sum())}; contains any of 235 inconsistent groups: {int((~clean[inb]).sum())}")
print(f"   dT between twins (all pairs): median |dT|={np.median(np.abs(Tm[pairs[:,0]]-Tm[pairs[:,1]]))/1e3:.2f} kW; p95={np.percentile(np.abs(Tm[pairs[:,0]]-Tm[pairs[:,1]]),95)/1e3:.1f} kW")
# L
res = []; res2 = []
for seed in range(20):
    rng = np.random.default_rng(seed); f = np.empty(U, dtype=int); f[rng.permutation(U)] = np.arange(U) % 5
    lk = np.zeros(U, dtype=bool); x = f[pairs[:, 0]] != f[pairs[:, 1]]; lk[pairs[x, 0]] = True; lk[pairs[x, 1]] = True; res.append(lk.mean() * 100)
    f = np.random.default_rng(1000 + seed).integers(0, 5, U); lk = np.zeros(U, dtype=bool); x = f[pairs[:, 0]] != f[pairs[:, 1]]; lk[pairs[x, 0]] = True; lk[pairs[x, 1]] = True; res2.append(lk.mean() * 100)
print(f"L: leak share over 20 seeds (balanced folds): mean={np.mean(res):.2f}% range {np.min(res):.2f}-{np.max(res):.2f}; (iid folds) mean={np.mean(res2):.2f}% range {np.min(res2):.2f}-{np.max(res2):.2f}")
# family-level: share of layouts whose FAMILY spans >1 fold is by construction different; skip
# M
k = 0
for (a, b2), m in zip(pairs, mv):
    ta, tb = sorted((Tm[a] / 1e3, Tm[b2] / 1e3))
    if abs(ta - 4135.9) < 0.06 and abs(tb - 4137.2) < 0.06:
        print(f"M: pair layouts(first rows0) {first[a]},{first[b2]}: moved {m:.2f} m, T {Tm[a]/1e3:.2f} / {Tm[b2]/1e3:.2f} kW, in best family: {bool(inb[a] and inb[b2])}"); k += 1
print("M: matching pairs found:", k, "; pairs in best family with move 74.5-75.5 m:", int(np.sum((mv >= 74.5) & (mv < 75.5) & inb[pairs[:, 0]])))
np.savez(os.path.join(d, "perth49_fam.npz"), fam=fam, first=first, Tm=Tm, pairs=pairs, mv=mv)

# Perth 100 row 850
X, Y, P, q, T, sc = load("Perth_100"); g2, f2 = groups(X, Y)
t = np.deg2rad(55.3)
uu = (X * np.cos(t) + Y * np.sin(t)) / sc
a = uu - uu.mean(1, keepdims=True); b = P - P.mean(1, keepdims=True); r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
pv = {}
for i in range(len(P)): pv.setdefault(P[i].tobytes(), set()).add(int(g2[i]))
print(f"\nPerth_100: corr(P,u) over rows: mean={r.mean():.3f} p1={np.percentile(r,1):.3f} min={r.min():.3f} (row0 {np.argmin(r)})")
for i in (848, 849, 850, 851):
    same = np.where(g2 == g2[i])[0]
    # nearest other layout by shared positions
    si = set(zip(X[i].tolist(), Y[i].tolist()))
    best = (0, -1)
    for j in f2:
        if g2[j] == g2[i]: continue
        sh = len(si & set(zip(X[j].tolist(), Y[j].tolist())))
        if sh > best[0]: best = (sh, j)
    j = best[1]; pj = dict(zip(zip(X[j].tolist(), Y[j].tolist()), P[j])); dd = np.array([P[i][n] - pj[k2] for n, k2 in enumerate(zip(X[i].tolist(), Y[i].tolist())) if k2 in pj])
    print(f" row0 {i}: T={T[i]:.1f} rank={int((T>T[i]).sum())+1} qW={q[i]} T-sum={T[i]-P[i].sum():.2f} corr(P,u)={r[i]:.3f} (pct {np.mean(r<r[i])*100:.1f}) Pmin/Pmax={P[i].min():.0f}/{P[i].max():.0f} dup rows={len(same)} powervec shared with other layouts={len(pv[P[i].tobytes()])-1}; nearest other layout row0 {j} shares {best[0]}/100 positions, T={T[j]:.1f}, |dP| at shared positions median={np.median(np.abs(dd)):.0f} max={np.abs(dd).max():.0f} W")
