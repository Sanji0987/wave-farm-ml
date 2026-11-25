#!/usr/bin/env python3
"""Pass 1 over the ERA5 Sri-Lanka-box file, in time chunks.
Per ERA5 cell: valid counts, time-mean wave power (0.49 Hs^2 Te, Te=mwp), plain mean mwp,
power-weighted mwp, mean Hs, mean pp1d etc.  Writes an .npz.
usage: 01_era5_stats.py <era5.nc> <out.npz>
"""
import sys
import numpy as np
import netCDF4 as nc

src, out = sys.argv[1], sys.argv[2]
d = nc.Dataset(src)
d.set_auto_mask(False)  # raw values; handle NaN / fill ourselves
t = d.variables["valid_time"][:]
lat = d.variables["latitude"][:]
lon = d.variables["longitude"][:]
print("vars:", [k for k in d.variables])
print("n time", len(t), "lat", lat[0], lat[-1], len(lat), "lon", lon[0], lon[-1], len(lon))
print("dlat set", np.unique(np.round(np.diff(lat), 6)), "dlon set", np.unique(np.round(np.diff(lon), 6)))
tt = np.asarray(t, dtype="int64")
print("time units:", d.variables["valid_time"].units)
print("first", np.datetime64(int(tt[0]), "s"), "last", np.datetime64(int(tt[-1]), "s"))
print("dt set (h):", np.unique(np.diff(tt)) / 3600.0)
try:
    ev = d.variables["expver"][:]
    print("expver unique:", np.unique(ev))
except Exception as e:  # noqa
    print("expver read failed", e)
for v in ["swh", "mwp", "pp1d", "mwd"]:
    V = d.variables[v]
    print(v, {a: V.getncattr(a) for a in V.ncattrs() if a in ("_FillValue", "missing_value", "units", "long_name", "GRIB_shortName", "GRIB_paramId", "GRIB_gridType")})

nt, ny, nx = len(t), len(lat), len(lon)
shape = (ny, nx)
acc = {k: np.zeros(shape, dtype="float64") for k in [
    "n_swh", "n_mwp", "n_pp1d", "n_mwd", "n_both", "n_all4",
    "sum_P", "sum_Pmwp", "sum_mwp", "sum_hs", "sum_hs2", "sum_pp1d", "sum_Ppp1d",
    "sum_ratio", "sum_P_pp1d857",
]}
CH = 650  # time steps per chunk (file chunking is 1950 x 34 x 34)
for i0 in range(0, nt, CH):
    i1 = min(nt, i0 + CH)
    hs = d.variables["swh"][i0:i1].astype("float64")
    te = d.variables["mwp"][i0:i1].astype("float64")
    tp = d.variables["pp1d"][i0:i1].astype("float64")
    md = d.variables["mwd"][i0:i1].astype("float64")
    # treat absurd fill values as missing too
    for a in (hs, te, tp, md):
        a[~np.isfinite(a)] = np.nan
        a[np.abs(a) > 1e6] = np.nan
    ok_h, ok_t, ok_p, ok_d = np.isfinite(hs), np.isfinite(te), np.isfinite(tp), np.isfinite(md)
    both = ok_h & ok_t
    acc["n_swh"] += ok_h.sum(0)
    acc["n_mwp"] += ok_t.sum(0)
    acc["n_pp1d"] += ok_p.sum(0)
    acc["n_mwd"] += ok_d.sum(0)
    acc["n_both"] += both.sum(0)
    acc["n_all4"] += (both & ok_p & ok_d).sum(0)
    P = 0.49 * hs * hs * te
    acc["sum_P"] += np.nansum(P, 0)
    acc["sum_Pmwp"] += np.nansum(P * te, 0)
    acc["sum_mwp"] += np.nansum(np.where(both, te, np.nan), 0)
    acc["sum_hs"] += np.nansum(np.where(both, hs, np.nan), 0)
    acc["sum_hs2"] += np.nansum(np.where(both, hs * hs, np.nan), 0)
    b3 = both & ok_p
    acc["sum_pp1d"] += np.nansum(np.where(b3, tp, np.nan), 0)
    acc["sum_Ppp1d"] += np.nansum(np.where(b3, P * tp, np.nan), 0)
    acc["sum_ratio"] += np.nansum(np.where(b3, tp / te, np.nan), 0)
    acc["sum_P_pp1d857"] += np.nansum(np.where(b3, 0.49 * hs * hs * 0.857 * tp, np.nan), 0)
    print("chunk", i0, i1, "done", flush=True)
d.close()
np.savez(out, lat=lat, lon=lon, time=tt, nt=nt, **acc)
n = acc["n_both"]
print("cells with any data:", int((n > 0).sum()))
print("cells with 100% data:", int((n == nt).sum()))
print("cells with >90% data:", int((n > 0.9 * nt).sum()))
print("cells with >=90% data:", int((n >= 0.9 * nt).sum()))
print("distinct n_both values:", np.unique(n)[:20], "...", np.unique(n)[-5:])
print("n_swh==n_mwp everywhere:", bool((acc["n_swh"] == acc["n_mwp"]).all()),
      " n_pp1d==n_swh:", bool((acc["n_pp1d"] == acc["n_swh"]).all()),
      " n_mwd==n_swh:", bool((acc["n_mwd"] == acc["n_swh"]).all()))
