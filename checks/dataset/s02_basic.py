#!/usr/bin/env python3
"""Claims A-G. Usage: python3 -I s02_basic.py <npz dir>"""
import sys, os
import numpy as np

d = sys.argv[1]
KEYS = ["Perth_49", "Sydney_49", "Perth_100", "Sydney_100"]
np.set_printoptions(linewidth=200, suppress=True)

def load(k):
    z = np.load(os.path.join(d, k + ".npz"))
    return z["X"], z["Y"], z["P"], z["q"], z["T"], int(z["scale"])

def ordered_groups(X, Y):
    """exact grouping on the ordered coordinate tuple (X1..XN,Y1..YN) using python dict of tuples (exact equality)."""
    n = X.shape[0]
    seen = {}
    gid = np.empty(n, dtype=np.int64)
    first = []
    for i in range(n):
        key = (X[i].tobytes(), Y[i].tobytes())  # exact bytes of int64 centimetres; dict compares bytes exactly
        g = seen.get(key)
        if g is None:
            g = len(first); seen[key] = g; first.append(i)
        gid[i] = g
    return gid, np.array(first)

def set_groups(X, Y):
    n, N = X.shape
    seen = {}
    gid = np.empty(n, dtype=np.int64); first = []
    for i in range(n):
        o = np.lexsort((Y[i], X[i]))
        key = (X[i][o].tobytes(), Y[i][o].tobytes())
        g = seen.get(key)
        if g is None:
            g = len(first); seen[key] = g; first.append(i)
        gid[i] = g
    return gid, np.array(first)

def min_pair_dist(Xm, Ym):
    """Xm,Ym in metres (n,N). returns min pair distance per layout, and count of pairs < 50 m"""
    n, N = Xm.shape
    iu = np.triu_indices(N, 1)
    mn = np.empty(n); cnt = np.empty(n, dtype=int)
    B = 500
    for s in range(0, n, B):
        dx = Xm[s:s+B, :, None] - Xm[s:s+B, None, :]
        dy = Ym[s:s+B, :, None] - Ym[s:s+B, None, :]
        D = np.sqrt(dx * dx + dy * dy)[:, iu[0], iu[1]]
        mn[s:s+B] = D.min(1); cnt[s:s+B] = (D < 50 - 1e-9).sum(1)
    return mn, cnt

tot_rows = 0; tot_uni = 0; tot_set = 0
store = {}
for k in KEYS:
    X, Y, P, q, T, sc = load(k)
    n, N = X.shape
    gid, first = ordered_groups(X, Y)
    sgid, sfirst = set_groups(X, Y)
    # cross-check with numpy unique (lexicographic exact comparison, no hashing)
    XY = np.concatenate([X, Y], axis=1)
    nu = np.unique(XY, axis=0).shape[0]
    assert nu == len(first), (nu, len(first))
    # full-row duplicates (coords + all powers + q + T)
    full = np.concatenate([XY.astype(np.float64), P, q[:, None], T[:, None]], axis=1)
    nfull = np.unique(full, axis=0).shape[0]
    u = len(first); su = len(sfirst)
    tot_rows += n; tot_uni += u; tot_set += su
    print(f"\n=== {k}: rows={n}  unique ordered layouts={u}  dup share={(n-u)/n*100:.2f}%  unique as SETS={su}  fully-identical-row uniques={nfull}")
    cnts = np.bincount(gid)
    print(f"   group sizes: max={cnts.max()}  groups with >1 row={np.sum(cnts>1)}  rows in those={cnts[cnts>1].sum()}")
    Xm = X / sc; Ym = Y / sc
    print(f"   X range {Xm.min()}..{Xm.max()}  Y range {Ym.min()}..{Ym.max()}")
    # duplicates of positions within a layout
    mn, cnt = min_pair_dist(Xm[first], Ym[first])
    print(f"   [unique layouts] min-pair-distance: min={mn.min():.3f} median={np.median(mn):.2f} max={mn.max():.2f}")
    print(f"   [unique layouts] share with a pair <50 m: {np.mean(mn < 50-1e-9)*100:.3f}%  ({np.sum(mn<50-1e-9)} of {u});  <=50: {np.sum(mn<=50)}; "
          f"with zero-distance pair: {np.sum(mn==0)}")
    for lo, hi in [(0, 6), (6, 10), (6, 10.0001), (5.999, 10.0001), (0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]:
        print(f"      min-dist in [{lo},{hi}): {np.sum((mn>=lo)&(mn<hi))}", end=";")
    print()
    mn_all, _ = min_pair_dist(Xm, Ym)
    print(f"   [all rows] share with a pair <50 m: {np.mean(mn_all<50-1e-9)*100:.3f}% ({np.sum(mn_all<50-1e-9)} of {n}); rows with min-dist in [6,10]: {np.sum((mn_all>=6)&(mn_all<=10))}")
    # set-unique variant
    mn_s = mn_all[sfirst]
    print(f"   [set-unique layouts] share with a pair <50 m: {np.mean(mn_s<50-1e-9)*100:.3f}% ({np.sum(mn_s<50-1e-9)} of {su})")
    # D: best/worst
    ib = np.argmax(T); iw = np.argmin(T)
    print(f"   best Total_Power={T[ib]:.2f} (row0 {ib}, qW={q[ib]}, per buoy {T[ib]/N:.2f});  worst={T[iw]:.2f} (row0 {iw}, qW={q[iw]})")
    # group mean power
    Tm = np.bincount(gid, weights=T) / cnts
    print(f"   over unique layouts (dup-averaged T): best={Tm.max():.2f} worst={Tm.min():.2f};  first-occurrence: best={T[first].max():.2f} worst={T[first].min():.2f}")
    # F: identical coords, different power
    Tmax = np.full(u, -np.inf); Tmin = np.full(u, np.inf)
    np.maximum.at(Tmax, gid, T); np.minimum.at(Tmin, gid, T)
    Pmax = np.full((u, N), -np.inf); Pmin = np.full((u, N), np.inf)
    np.maximum.at(Pmax, gid, P); np.minimum.at(Pmin, gid, P)
    qmax = np.full(u, -np.inf); qmin = np.full(u, np.inf)
    np.maximum.at(qmax, gid, q); np.minimum.at(qmin, gid, q)
    diffT = (Tmax - Tmin) > 0
    diffP = ((Pmax - Pmin) > 0).any(1)
    diffany = diffT | diffP | ((qmax - qmin) > 0)
    print(f"   F: groups with differing Total_Power: {diffT.sum()} (rows {cnts[diffT].sum()});  differing any Power_i: {diffP.sum()} (rows {cnts[diffP].sum()}); "
          f"differing any of P/q/T: {diffany.sum()} (rows {cnts[diffany].sum()})")
    if diffany.sum():
        sprT = (Tmax - Tmin)[diffany]
        sprP = (Pmax - Pmin)[diffany].max(1)
        print(f"      Total_Power spread within group: max={sprT.max():.2f} W (= {sprT.max()/N:.2f} W per device), median={np.median(sprT):.2f}")
        print(f"      per-buoy Power_i spread within group: max={sprP.max():.2f} W, median of group-max={np.median(sprP):.2f}")
        # sd-based
        # how far could averaging move group value from any single row
        dev = np.abs(T - Tm[gid])
        print(f"      max |row T - group-mean T| = {dev.max():.2f} W ; mean abs dev over rows in differing groups = {dev[diffany[gid]].mean():.2f} W")
        gb = gid[ib]; gw = gid[iw]
        print(f"      best row's group: size={cnts[gb]} Tmin={Tmin[gb]:.2f} Tmax={Tmax[gb]:.2f} mean={Tm[gb]:.2f};  worst row's group: size={cnts[gw]} Tmin={Tmin[gw]:.2f} Tmax={Tmax[gw]:.2f} mean={Tm[gw]:.2f}")
        # top-10 unique by mean vs by first
        # groups by size among differing
        print(f"      differing groups size distribution: min={cnts[diffany].min()} median={np.median(cnts[diffany])} max={cnts[diffany].max()}")
    # G: total vs sum of powers
    S = P.sum(1)
    mm = np.abs(T - S)
    im = np.argmax(mm)
    print(f"   G: max |Total_Power - sum(Power_i)| = {mm.max():.2f} W at row0 {im} (T={T[im]:.2f}, sum={S[im]:.2f});  rows with mismatch >1 W: {np.sum(mm>1)}; >100 W: {np.sum(mm>100)}; >1 kW: {np.sum(mm>1000)}; median={np.median(mm):.4f}")
    print(f"      signed diff T-sum: min={np.min(T-S):.2f} max={np.max(T-S):.2f}")
    # E: P_iso bounds
    r = T / (N * q)
    lo = T / (N * (q + 0.005)); hi = T / (N * (q - 0.005))
    print(f"   E: naive ratio T/(N*qW): mean={r.mean():.2f} median={np.median(r):.2f} min={r.min():.2f} max={r.max():.2f}")
    print(f"      round-to-nearest hypothesis: P_iso in [{lo.max():.3f}, {hi.min():.3f}]  (non-empty={lo.max()<=hi.min()})")
    # using sum(Power_i) instead
    lo2 = S / (N * (q + 0.005)); hi2 = S / (N * (q - 0.005))
    print(f"      same using sum(Power_i): [{lo2.max():.3f}, {hi2.min():.3f}] (non-empty={lo2.max()<=hi2.min()})")
    # truncation hypothesis
    lo3 = T / (N * (q + 0.01)); hi3 = T / (N * q)
    print(f"      truncation hypothesis: [{lo3.max():.3f}, {hi3.min():.3f}] (non-empty={lo3.max()<=hi3.min()})")
    store[k] = dict(lo=lo, hi=hi, T=T, q=q, N=N)

print(f"\nTOTAL rows={tot_rows} unique ordered={tot_uni} unique sets={tot_set}")
for site in ["Perth", "Sydney"]:
    a = store[site + "_49"]; b = store[site + "_100"]
    lo = max(a["lo"].max(), b["lo"].max()); hi = min(a["hi"].min(), b["hi"].min())
    print(f"{site}: joint 49+100 round-to-nearest interval [{lo:.3f}, {hi:.3f}] non-empty={lo<=hi}")
