#!/usr/bin/env python3
"""Band ratio stats ignoring all-NaN pp1d cells; D coincidence check. usage: 05b_fix.py <band_ts.npz> <cells_with_wave.pkl>"""
import sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
z = np.load(sys.argv[1]); df = pd.read_pickle(sys.argv[2])
cells = z["cells"]; col = {c: k for k, c in enumerate(cells)}
hs, te, tp = (z[v].astype("float64") for v in ("swh", "mwp", "pp1d"))
R = tp / te
allnan = np.isnan(tp).all(0)
part = np.isnan(tp).any(0) & ~allnan
print("ERA5 band cells with pp1d entirely missing:", int(allnan.sum()), " partly missing:", int(part.sum()),
      " partly-missing share range: %.3f-%.3f" % (np.isnan(tp[:, part]).mean(0).min(), np.isnan(tp[:, part]).mean(0).max()) if part.any() else "")
bc = df.era_flat[df.band & df.has].values.astype(int)
print("coastal band cells mapped to all-NaN-pp1d ERA5 cells:", int(allnan[[col[c] for c in bc]].sum()))
cmed = np.nanmedian(R, 0)
v = cmed[~allnan]
print("per-ERA5-cell median ratio (n=%d): median %.3f p5 %.3f p25 %.3f p75 %.3f p95 %.3f min %.3f max %.3f" % (
    len(v), np.median(v), *np.percentile(v, [5, 25, 75, 95]), v.min(), v.max()))
pc = cmed[[col[c] for c in bc]]; pc = pc[np.isfinite(pc)]
print("per-coastal-cell median ratio (n=%d): median %.3f p5 %.3f p25 %.3f p75 %.3f p95 %.3f" % (len(pc), np.median(pc), *np.percentile(pc, [5, 25, 75, 95])))
# D: precision
j = ((df.lat - 6.5083).abs() + (df.lon - 93.8083).abs()).idxmin(); c6 = col[int(df.loc[j, "era_flat"])]
a = np.mean(0.49 * hs[:, c6] ** 2 * te[:, c6]) / 0.857
b = np.mean(0.49 * hs[:, c6] ** 2 * 0.857 * tp[:, c6])
print("D: P6/0.857 = %.6f ; mean 0.49 Hs2 (0.857 pp1d) = %.6f" % (a, b))
