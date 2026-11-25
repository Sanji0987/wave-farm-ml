#!/usr/bin/env python3
"""E (pp1d/mwp ratios), D (EC1 factor), G (mwd sectors at site 1).
usage: 05_periods_dirs.py <band_ts.npz> <cells_with_wave.pkl> <era5_stats.npz>
"""
import sys
import numpy as np
import pandas as pd

z = np.load(sys.argv[1])
df = pd.read_pickle(sys.argv[2])
st = np.load(sys.argv[3])
cells = z["cells"]
col = {c: k for k, c in enumerate(cells)}
hs, te, tp, md = (z[v].astype("float64") for v in ("swh", "mwp", "pp1d", "mwd"))
P = 0.49 * hs * hs * te
R = tp / te
sites = [(7.6083, 77.9583), (8.0083, 78.2583), (7.5583, 77.1583), (6.6083, 81.9583), (6.0083, 81.4083), (6.5083, 93.8083)]
sc = []
for la, lo in sites:
    j = ((df.lat - la).abs() + (df.lon - lo).abs()).idxmin()
    sc.append(col[int(df.loc[j, "era_flat"])])


def q(a, p):
    return np.nanpercentile(a, p)


print("--- E: per-step ratio pp1d/mwp at site ERA5 cells")
print("site  n_pp1d  med  p25  p75  p10  p90 | mean mwp  mean mwp/.857  mean pp1d  Hs2w mwp/.857  Hs2w pp1d  share(pp1d>mwp/.857) | share pp1d>12s")
for k, c in enumerate(sc, 1):
    r = R[:, c]
    ok = np.isfinite(r)
    w = hs[:, c] ** 2
    a, b = te[ok, c] / 0.857, tp[ok, c]
    print("%d %5d  %.3f %.3f %.3f %.3f %.3f | %.2f %.2f %.2f  %.2f %.2f  %.3f | %.3f" % (
        k, ok.sum(), q(r, 50), q(r, 25), q(r, 75), q(r, 10), q(r, 90), np.mean(te[:, c]), np.mean(te[:, c]) / 0.857,
        np.nanmean(tp[:, c]), np.sum(w[ok] * a) / np.sum(w[ok]), np.sum(w[ok] * b) / np.sum(w[ok]), np.mean(b > a),
        np.mean(b > 12)))
# implied Te/Tp
print("implied Te/Tp (=1/median ratio) at site1: %.3f, site6: %.3f" % (1 / q(R[:, sc[0]], 50), 1 / q(R[:, sc[5]], 50)))

# band: weight by coastal band cell (each coastal cell -> its ERA5 cell)
bandcells = df.era_flat[df.band & df.has].values.astype(int)
cmed = np.nanmedian(R, axis=0)  # per ERA5 cell
cmean = np.nanmean(R, axis=0)
per_coastal = cmed[[col[c] for c in bandcells]]
print("\n--- E: band (5,433 coastal cells -> 556 ERA5 cells)")
print("per-ERA5-cell median ratio across 556 cells: median %.3f, p5 %.3f, p25 %.3f, p75 %.3f, p95 %.3f, min %.3f, max %.3f" % (
    np.median(cmed), q(cmed, 5), q(cmed, 25), q(cmed, 75), q(cmed, 95), cmed.min(), cmed.max()))
print("per-coastal-cell median ratio: median %.3f, p5 %.3f, p95 %.3f" % (np.median(per_coastal), q(per_coastal, 5), q(per_coastal, 95)))
pooled = R[:, [col[c] for c in np.unique(bandcells)]].ravel()
print("pooled per-step ratio (556 cells): median %.3f p25 %.3f p75 %.3f p10 %.3f p90 %.3f; share <1.167: %.3f" % (
    q(pooled, 50), q(pooled, 25), q(pooled, 75), q(pooled, 10), q(pooled, 90), np.nanmean(pooled < 1 / 0.857)))
print("cells whose median ratio < 1/0.857=1.167:", int((cmed < 1 / 0.857).sum()), "of", len(cmed))
print("pp1d missing share at band ERA5 cells: %.4f; cells with any pp1d missing: %d" % (np.isnan(tp).mean(), int(np.isnan(tp).any(0).sum())))
for k, c in enumerate(sc, 1):
    print("  site %d pp1d missing steps: %d" % (k, int(np.isnan(tp[:, c]).sum())))

print("\n--- D: EC1 factor at site 6")
P6 = np.mean(P[:, sc[5]])
print("P6 = %.4f ; /0.857 = %.3f ; x1.10 = %.3f ; x1.12 = %.3f ; x1.14 = %.3f ; x1.17 = %.3f" % (P6, P6 / .857, P6 * 1.1, P6 * 1.12, P6 * 1.14, P6 * 1.17))
c6 = sc[5]
ok = np.isfinite(tp[:, c6])
print("0.49*Hs2*(0.857*pp1d) mean at site 6: %.3f ; 0.49*Hs2*pp1d: %.3f" % (np.mean(0.49 * hs[ok, c6] ** 2 * 0.857 * tp[ok, c6]), np.mean(0.49 * hs[ok, c6] ** 2 * tp[ok, c6])))
print("0.5*Hs2*mwp: %.3f ; 0.5*Hs2*mwp/0.857: %.3f ; 0.49*meanHs^2*mean Te: %.3f" % (np.mean(0.5 * hs[:, c6] ** 2 * te[:, c6]), np.mean(0.5 * hs[:, c6] ** 2 * te[:, c6]) / .857, 0.49 * np.mean(hs[:, c6]) ** 2 * np.mean(te[:, c6])))
# band max under /0.857
allP = np.mean(P, axis=0)
print("band max cell P /0.857 = %.3f; site1 /0.857 = %.3f" % (allP.max() / .857, np.mean(P[:, sc[0]]) / .857))

print("\n--- G: mwd at site 1 (direction waves come FROM, deg true)")
c1 = sc[0]
d1, p1 = md[:, c1], P[:, c1]
edges = np.arange(0, 361, 30)
h, _ = np.histogram(d1, edges)
hp, _ = np.histogram(d1, edges, weights=p1)
for a, b, n, pw in zip(edges[:-1], edges[1:], h, hp):
    print("  %3d-%3d: time %.3f power %.3f" % (a, b, n / len(d1), pw / p1.sum()))
# best two-sector split: half-circle [th, th+180) vs rest, choose th minimising steps within +-20 deg of the two cut lines
best = None
for th in range(0, 180):
    lo = (d1 - th) % 360
    near = np.minimum(np.minimum(lo, 360 - lo), np.abs(lo - 180)) < 20
    s = near.mean()
    if best is None or s < best[0]:
        best = (s, th)
th = best[1]
inA = ((d1 - th) % 360) < 180
tA, pA = inA.mean(), p1[inA].sum() / p1.sum()
print("valley split at %d / %d deg (%.3f of steps within 20 deg of cuts)" % (th, th + 180, best[0]))
print("  sector [%d,%d): time %.3f power %.3f ; other: time %.3f power %.3f" % (th, th + 180, tA, pA, 1 - tA, 1 - pA))
# fixed split N-S axis halves: from east (0-180) vs from west (180-360)
inE = (d1 >= 0) & (d1 < 180)
print("  from east half (0-180): time %.3f power %.3f" % (inE.mean(), p1[inE].sum() / p1.sum()))
# circular mean of power-weighted dir and +-90 half
ang = np.radians(d1)
mdir = np.degrees(np.arctan2(np.sum(p1 * np.sin(ang)), np.sum(p1 * np.cos(ang)))) % 360
dev = np.abs((d1 - mdir + 180) % 360 - 180)
print("  power-weighted mean dir %.1f; >90 deg away: time %.3f power %.3f" % (mdir, (dev > 90).mean(), p1[dev > 90].sum() / p1.sum()))
# by month
t = st["time"]
mon = (t.astype("datetime64[s]").astype("datetime64[M]").astype(int) % 12) + 1
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    m = mon == k
    print("  month %2d: median mwd %.0f, share in sector[%d,%d) %.2f, mean P %.1f" % (k, np.median(d1[m]), th, th + 180, inA[m].mean(), p1[m].mean()))
# same at site 6 for comparison
c6 = sc[5]
d6, p6 = md[:, c6], P[:, c6]
h6, _ = np.histogram(d6, edges, weights=p6)
print("site 6 power share by 30deg bin:", np.round(h6 / p6.sum(), 3))
