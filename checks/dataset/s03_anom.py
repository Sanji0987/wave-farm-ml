#!/usr/bin/env python3
"""Explore F/G anomalies. Usage: python3 -I s03_anom.py <npz dir>"""
import sys, os
import numpy as np
d = sys.argv[1]
np.set_printoptions(linewidth=220, suppress=True, precision=1)

def load(k):
    z = np.load(os.path.join(d, k + ".npz"))
    return z["X"], z["Y"], z["P"], z["q"], z["T"], int(z["scale"])

def groups(X, Y):
    seen = {}; gid = np.empty(len(X), dtype=np.int64); first = []
    for i in range(len(X)):
        key = (X[i].tobytes(), Y[i].tobytes()); g = seen.get(key)
        if g is None:
            g = len(first); seen[key] = g; first.append(i)
        gid[i] = g
    return gid, np.array(first)

# exact integer check of the 50 m rule
for k in ["Perth_49", "Sydney_49", "Perth_100", "Sydney_100"]:
    X, Y, P, q, T, sc = load(k)
    gid, first = groups(X, Y)
    Xi = X[first]; Yi = Y[first]
    N = X.shape[1]; iu = np.triu_indices(N, 1)
    mn2 = np.empty(len(first), dtype=np.int64)
    for s in range(0, len(first), 400):
        dx = Xi[s:s+400, :, None] - Xi[s:s+400, None, :]; dy = Yi[s:s+400, :, None] - Yi[s:s+400, None, :]
        D2 = (dx * dx + dy * dy)[:, iu[0], iu[1]]
        mn2[s:s+400] = D2.min(1)
    lim = (50 * sc) ** 2
    print(f"{k}: exact-integer: unique layouts with pair <50 m: {np.sum(mn2<lim)} / {len(first)} = {np.mean(mn2<lim)*100:.3f}% ; ==50 m exactly: {np.sum(mn2==lim)}; "
          f"min-dist 6..10 m: {np.sum((mn2>=(6*sc)**2)&(mn2<=(10*sc)**2))}; <6 m: {np.sum(mn2<(6*sc)**2)}; min={np.sqrt(mn2.min())/sc:.3f} m")
    if k == "Perth_49":
        v = np.sort(np.sqrt(mn2[mn2 < lim]) / sc)
        print("   Perth49 sub-50 min-distances (m):", np.round(v, 2))

print("\n===== Perth_49 Total vs sum mismatch structure =====")
X, Y, P, q, T, sc = load("Perth_49")
gid, first = groups(X, Y)
cnts = np.bincount(gid)
S = P.sum(1); mm = T - S
big = np.abs(mm) > 1.0
print("rows with |T-sum|>1 W:", big.sum(), " distinct layouts among them:", len(np.unique(gid[big])))
print("row index range of mismatching rows:", np.where(big)[0].min(), np.where(big)[0].max())
# histogram of mismatch rows over file position
h, e = np.histogram(np.where(big)[0], bins=np.arange(0, 37001, 2000))
print("mismatch rows per 2000-row block:", h)
print("quantiles of T-sum among mismatch rows:", np.percentile(mm[big], [0, 5, 25, 50, 75, 95, 100]))
# relation with the differing groups
u = len(first)
Tmax = np.full(u, -np.inf); Tmin = np.full(u, np.inf)
np.maximum.at(Tmax, gid, T); np.minimum.at(Tmin, gid, T)
dg = (Tmax - Tmin) > 0
print("differing groups:", dg.sum(), "rows:", cnts[dg].sum())
print("mismatch rows that are in differing groups:", (big & dg[gid]).sum(), " mismatch rows not in differing groups:", (big & ~dg[gid]).sum())
print("rows in differing groups without mismatch:", (~big & dg[gid]).sum())
# file position of rows in differing groups
h2, _ = np.histogram(np.where(dg[gid])[0], bins=np.arange(0, 37001, 2000))
print("rows in differing groups per 2000-row block:", h2)
# number of distinct T values per differing group
nd = np.array([len(np.unique(T[gid == g])) for g in np.where(dg)[0]])
print("distinct Total_Power values per differing group: ", np.bincount(nd))
# show the group with largest spread
spr = Tmax - Tmin
g = np.argmax(spr)
rows = np.where(gid == g)[0]
print(f"\nlargest-spread group g={g}: size={len(rows)} rows(0-based)={rows[:20]}... T values (unique, counts):")
uu, cc = np.unique(T[rows], return_counts=True)
for a, b in zip(uu, cc):
    r0 = rows[T[rows] == a][0]
    print(f"   T={a:.2f} x{b}  qW={q[r0]}  sum(P)={S[r0]:.2f}  first row0={r0}")
# are the differing power vectors permutations of each other?
r_hi = rows[np.argmax(T[rows])]; r_lo = rows[np.argmin(T[rows])]
print("  P (highest T row):", P[r_hi][:12]); print("  P (lowest T row): ", P[r_lo][:12])
print("  sorted P equal (permutation)?", np.allclose(np.sort(P[r_hi]), np.sort(P[r_lo])))
# Is a differing row's power vector identical to the power vector of some OTHER layout's row?
# build dict of power-vector -> set of group ids
pv = {}
for i in range(len(P)):
    pv.setdefault(P[i].tobytes(), set()).add(int(gid[i]))
multi = {k_: v for k_, v in pv.items() if len(v) > 1}
print("\ndistinct per-buoy power vectors:", len(pv), "; power vectors shared by >1 distinct coordinate layout:", len(multi))
rows_multi = sum(1 for i in range(len(P)) if len(pv[P[i].tobytes()]) > 1)
print("rows whose power vector also appears under different coordinates:", rows_multi)
# same on (T) only
tv = {}
for i in range(len(T)):
    tv.setdefault(float(T[i]), set()).add(int(gid[i]))
print("distinct Total_Power values:", len(tv), "; T values shared by >1 distinct layout:", sum(1 for v in tv.values() if len(v) > 1))
# within differing groups, time order: does T increase / change along file order?
print("\nexample differing groups (first 6): file rows and T")
for g in np.where(dg)[0][:6]:
    rows = np.where(gid == g)[0]
    print(f"  g={g} n={len(rows)} rows={rows[:8]} T={np.round(T[rows][:8],1)} qW={q[rows][:8]}")
# how many differing groups are among top layouts?
Tm = np.bincount(gid, weights=T) / cnts
order = np.argsort(-Tm)
rank = np.empty(u, dtype=int); rank[order] = np.arange(u)
print("\nrank (by mean T) of differing groups: quantiles", np.percentile(rank[dg], [0, 10, 25, 50, 75, 90, 100]))
print("differing groups in top-5% (by mean):", np.sum(rank[dg] < 0.05 * u), " in top-1%:", np.sum(rank[dg] < 0.01 * u))
# effect of averaging on the distribution of unique-layout T
Tf = T[first]
print(f"unique-layout T: mean (dup-avg)={Tm.mean():.2f}, mean (first occ)={Tf.mean():.2f}, mean(max)={Tmax.mean():.2f}, mean(min)={Tmin.mean():.2f}; std avg={Tm.std():.2f} first={Tf.std():.2f}")
print(f"max |Tm-Tfirst|={np.abs(Tm-Tf).max():.2f}; top-10 by mean: {np.round(np.sort(Tm)[-10:],1)}; top-10 by first: {np.round(np.sort(Tf)[-10:],1)}")
print(f"second-best unique layout: avg={np.sort(Tm)[-2]:.2f} first={np.sort(Tf)[-2]:.2f} max={np.sort(Tmax)[-2]:.2f}")

# best row details
ib = np.argmax(T)
gb = gid[ib]; rows = np.where(gid == gb)[0]
print(f"\nBEST row0={ib}: T={T[ib]:.2f} sum(P)={S[ib]:.2f} diff={T[ib]-S[ib]:.2f} qW={q[ib]}; group size={len(rows)}, rows {rows.min()}..{rows.max()}; distinct sum(P) in group: {np.unique(np.round(S[rows],2))}")
print("  all P identical within group:", np.all(P[rows] == P[rows[0]]))
# top 10 rows by T: their mismatch
top = np.argsort(-Tm)[:10]
for g in top:
    r = np.where(gid == g)[0]
    print(f"   layout g={g} size={len(r)} T={np.unique(T[r])} sumP={np.unique(np.round(S[r],2))} qW={np.unique(q[r])}")

print("\n===== per-file mismatch detail =====")
for k in ["Perth_49", "Sydney_49", "Perth_100", "Sydney_100"]:
    X, Y, P, q, T, sc = load(k)
    S = P.sum(1); mm = T - S
    o = np.argsort(-np.abs(mm))[:5]
    print(k, "top-5 |T-sum| rows0:", o, np.round(mm[o], 2), " T:", np.round(T[o], 1))
    print("   best row0:", np.argmax(T), "mismatch at best row:", round(float(mm[np.argmax(T)]), 2))
    # does max(sum P) differ from max(T)?
    print(f"   max sum(P)={S.max():.2f} at row0 {np.argmax(S)};  max T={T.max():.2f} at row0 {np.argmax(T)}")
