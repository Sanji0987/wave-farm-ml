#!/usr/bin/env python3
"""Try matching variants to see which gives the team's 5,422 band cells with wave data.
usage: 03_variants.py <cells_with_wave.pkl> <era5_stats.npz>
"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

df = pd.read_pickle(sys.argv[1])
z = np.load(sys.argv[2])
lat, lon, nt = z["lat"], z["lon"], int(z["nt"])
LO, LA = np.meshgrid(lon, lat)
good = (z["n_both"] > 0.9 * nt).ravel()
band = df.band.values
R = 6371.0088


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def hav(lat1, lon1, lat2, lon2, R=R):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


bl, bo = df.lat.values[band], df.lon.values[band]
print("baseline (nearest good cell, haversine<=20):", int((df.era_km.values[band] <= 20).sum()))
print("strict <20:", int((df.era_km.values[band] < 20).sum()))
# V2: nearest of ALL ERA5 cells, accept only if that cell is good and within 20 km
tall = cKDTree(xyz(LA.ravel(), LO.ravel()))
ch, ia = tall.query(xyz(bl, bo))
dk = 2 * R * np.arcsin(ch / 2)
print("V2 nearest-any-cell must have data, <=20 km:", int((good[ia] & (dk <= 20)).sum()),
      " (nearest-any has data:", int(good[ia].sum()), ")")
# V3: equirectangular degrees*111 km
gi = np.flatnonzero(good)
for scale, nm in [(111.0, "111"), (111.32, "111.32"), (111.195, "111.195")]:
    t = cKDTree(np.c_[LA.ravel()[gi], LO.ravel()[gi]])
    dd, _ = t.query(np.c_[bl, bo])
    print("V3 plain-degree dist*%s <=20:" % nm, int((dd * scale <= 20).sum()))
    # with cos(lat) scaling
t = cKDTree(np.c_[LA.ravel()[gi], LO.ravel()[gi] * np.cos(np.radians(10))])
# V4: nearest by degrees, then haversine
t = cKDTree(np.c_[LA.ravel()[gi], LO.ravel()[gi]])
dd, ii = t.query(np.c_[bl, bo])
dh = hav(bl, bo, LA.ravel()[gi][ii], LO.ravel()[gi][ii])
print("V4 nearest by degrees, haversine<=20:", int((dh <= 20).sum()))
# V5: R = 6371 / 6378
for RR in (6371.0, 6378.137):
    dh2 = df.era_km.values[band] * RR / R
    print("V5 R=%g <=20:" % RR, int((dh2 <= 20).sum()))
# V6: data threshold variants irrelevant (all-or-nothing). V7: P finite & Te finite after 20 km
# V8: band defined on wave-data cells with 0<P
# V9: maybe a 0.2 deg box test
print("V9 |dlat|<=0.2 & |dlon|<=0.2 (box) to nearest good:", int(((np.abs(bl - df.era_lat.values[band]) <= 0.2) & (np.abs(bo - df.era_lon.values[band]) <= 0.2)).sum()))
print("V10 within 0.18 deg euclid:", int((dd <= 0.18).sum()), " 0.2:", int((dd <= 0.2).sum()))
# distribution of distances near 20
e = df.era_km.values[band]
print("band cells with 19<d<=20:", int(((e > 19) & (e <= 20)).sum()), " 20<d<=21:", int(((e > 20) & (e <= 21)).sum()))
# which band cells are in 19.5-20 km?
print("band cells 19.8<d<=20:", int(((e > 19.8) & (e <= 20)).sum()))
