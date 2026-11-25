#!/usr/bin/env python3
"""Extract time series (swh, mwp, pp1d, mwd) at the ERA5 cells used by band cells.
usage: 04_extract_ts.py <era5.nc> <cells_with_wave.pkl> <out.npz>
"""
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

src, pkl, out = sys.argv[1:4]
df = pd.read_pickle(pkl)
cells = np.unique(df.era_flat[df.band & df.has].values.astype(int))
d = nc.Dataset(src)
d.set_auto_mask(False)
ny, nx = len(d.variables["latitude"]), len(d.variables["longitude"])
iy, ix = np.unravel_index(cells, (ny, nx))
y0, y1, x0, x1 = iy.min(), iy.max() + 1, ix.min(), ix.max() + 1
print("cells", len(cells), "bbox rows", y0, y1, "cols", x0, x1)
nt = len(d.variables["valid_time"])
res = {v: np.empty((nt, len(cells)), dtype="float32") for v in ["swh", "mwp", "pp1d", "mwd"]}
CH = 650
for i0 in range(0, nt, CH):
    i1 = min(nt, i0 + CH)
    for v in res:
        a = d.variables[v][i0:i1, y0:y1, x0:x1]
        res[v][i0:i1] = a[:, iy - y0, ix - x0]
    print("chunk", i0, flush=True)
d.close()
np.savez(out, cells=cells, **res)
print("nan share pp1d:", float(np.isnan(res["pp1d"]).mean()), " swh:", float(np.isnan(res["swh"]).mean()))
