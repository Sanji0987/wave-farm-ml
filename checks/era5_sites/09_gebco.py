#!/usr/bin/env python3
"""C: GEBCO depth at the six sites (point and mean over the 0.05-deg cell). usage: 09_gebco.py <gebco.nc>"""
import sys
import numpy as np, netCDF4 as nc
d = nc.Dataset(sys.argv[1]); la = d.variables["lat"][:]; lo = d.variables["lon"][:]; E = d.variables["elevation"]
print("gebco grid", len(la), len(lo), "dlat", la[1]-la[0])
sites = [(7.6083, 77.9583), (8.0083, 78.2583), (7.5583, 77.1583), (6.6083, 81.9583), (6.0083, 81.4083), (6.5083, 93.8083)]
for k, (a, b) in enumerate(sites, 1):
    i, j = int(np.argmin(np.abs(la-a))), int(np.argmin(np.abs(lo-b)))
    h = 6  # +-6 cells of 15" = +-0.025 deg -> 0.05-deg box
    blk = np.asarray(E[i-h:i+h, j-h:j+h], dtype=float)
    print("site %d: point depth %.0f m ; 0.05-deg box mean %.0f, median %.0f, min..max %.0f..%.0f" % (
        k, -float(E[i, j]), -blk.mean(), -np.median(blk), -blk.max(), -blk.min()))
