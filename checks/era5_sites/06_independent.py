#!/usr/bin/env python3
"""F: greedy selection of band cells >= 40 km apart. usage: 06_independent.py <cells_with_wave.pkl>"""
import sys
import numpy as np, pandas as pd
df = pd.read_pickle(sys.argv[1]); R = 6371.0088
def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
def greedy(sub, order, dmin=40.0):
    X = xyz(sub.lat.values, sub.lon.values)[order]
    thr = 2*np.sin(dmin/(2*R))  # chord
    acc = []
    for i in range(len(X)):
        if not acc or np.min(np.linalg.norm(X[acc] - X[i], axis=1)) >= thr:
            acc.append(i)
    return len(acc)
# degree-distance matched set (team-style, 5,422)
dd = np.hypot(df.lat - df.era_lat, df.lon - df.era_lon) * 111.0
sets = {"band+wave (haversine, 5433)": df.band & df.has,
        "band+wave (degree dist, 5422)": df.band & (dd <= 20),
        "band+wave unflagged": df.band & df.has & ~df.flag_b,
        "band all (5663)": df.band}
for nm, m in sets.items():
    sub = df[m].reset_index(drop=True)
    o = np.argsort(-sub.P.fillna(-1).values, kind="stable")
    res = [greedy(sub, o)]
    rnd = [greedy(sub, np.random.default_rng(s).permutation(len(sub))) for s in range(20)]
    # also ascending P and lat/lon raster order
    asc = greedy(sub, np.argsort(sub.P.fillna(99).values, kind="stable"))
    ras = greedy(sub, np.lexsort((sub.lon.values, sub.lat.values)))
    print("%-32s n=%5d  descP=%d  ascP=%d  raster=%d  random(20 seeds) min %d med %.0f max %d" % (
        nm, len(sub), res[0], asc, ras, min(rnd), np.median(rnd), max(rnd)))
# distinct ERA5 cells used by band, 40 km apart
e = df[df.band & df.has].drop_duplicates("era_flat")
sub = pd.DataFrame({"lat": e.era_lat.values, "lon": e.era_lon.values, "P": e.P.values})
print("distinct ERA5 cells n=%d descP=%d" % (len(sub), greedy(sub, np.argsort(-sub.P.values))))
for dmin in (30, 50):
    s2 = df[df.band & df.has].reset_index(drop=True)
    print("dmin %d km, band+wave descP: %d" % (dmin, greedy(s2, np.argsort(-s2.P.values, kind="stable"), dmin)))
