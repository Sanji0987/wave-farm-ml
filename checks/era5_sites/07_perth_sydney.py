#!/usr/bin/env python3
"""I: wave power in the small Perth/Sydney ERA5 boxes. usage: 07_perth_sydney.py <nc> [<nc> ...]"""
import sys
import numpy as np, netCDF4 as nc
for p in sys.argv[1:]:
    d = nc.Dataset(p); d.set_auto_mask(False)
    la, lo = d.variables["latitude"][:], d.variables["longitude"][:]
    hs = d.variables["swh"][:].astype("f8"); te = d.variables["mwp"][:].astype("f8"); tp = d.variables["pp1d"][:].astype("f8")
    t = d.variables["valid_time"][:]
    d.close()
    n = np.isfinite(hs).sum(0); nt = hs.shape[0]
    P = np.nanmean(0.49*hs*hs*te, 0); P[n < 0.9*nt] = np.nan
    print("%s: lat %.2f..%.2f lon %.2f..%.2f (%dx%d), %d steps %s..%s, cells with >90%% data: %d" % (
        p.split("/")[-1], la.min(), la.max(), lo.min(), lo.max(), len(la), len(lo), nt,
        np.datetime64(int(t[0]), "s"), np.datetime64(int(t[-1]), "s"), int((n > 0.9*nt).sum())))
    print("   P kW/m over data cells: min %.1f median %.1f max %.1f ; box mean %.1f" % (np.nanmin(P), np.nanmedian(P), np.nanmax(P), np.nanmean(P)))
    print("   mean Hs %.2f, mean mwp %.2f, mean pp1d %.2f (data cells)" % (np.nanmean(hs), np.nanmean(te), np.nanmean(tp)))
    # per-row summary (lat) of P to show the gradient; westmost/eastmost data cell
    with np.printoptions(precision=1, suppress=True, linewidth=150, nanstr="  -"):
        print("   lon:", lo); 
        for i in range(len(la)): print("   %7.2f" % la[i], P[i])
