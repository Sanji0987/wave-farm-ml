#!/usr/bin/env python3
"""Map msr_cells to ERA5 cells, band stats, six sites, flagged share, ranking.
usage: 02_cells.py <msr_cells.csv> <era5_stats.npz> <out_dir>
"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

csv, npz, od = sys.argv[1:4]
R = 6371.0088


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


z = np.load(npz)
lat, lon, nt = z["lat"], z["lon"], int(z["nt"])
LO, LA = np.meshgrid(lon, lat)
n = z["n_both"]
good = n > 0.9 * nt
P = np.where(good, z["sum_P"] / np.maximum(n, 1), np.nan)
Te_plain = np.where(good, z["sum_mwp"] / np.maximum(n, 1), np.nan)
Te_pw = np.where(good, z["sum_Pmwp"] / np.maximum(z["sum_P"], 1e-12), np.nan)
Hs = np.where(good, z["sum_hs"] / np.maximum(n, 1), np.nan)
Hrms2 = np.where(good, z["sum_hs2"] / np.maximum(n, 1), np.nan)
npp = z["n_pp1d"]
Tp_mean = np.where(good, z["sum_pp1d"] / np.maximum(npp, 1), np.nan)
ratio_mean = np.where(good, z["sum_ratio"] / np.maximum(npp, 1), np.nan)
print("ERA5 good cells:", int(good.sum()), " P range kW/m:", np.nanmin(P), np.nanmax(P))

df = pd.read_csv(csv)
print("msr rows:", len(df), " cols:", len(df.columns))
print("flagged dtype:", df["flagged"].dtype, df["flagged"].value_counts(dropna=False).to_dict())
# grid spacing
ul, uo = np.unique(np.round(df.lat, 5)), np.unique(np.round(df.lon, 5))
print("lat spacing set:", np.unique(np.round(np.diff(ul), 4))[:5], " lon spacing set:", np.unique(np.round(np.diff(uo), 4))[:5])
print("lat range", df.lat.min(), df.lat.max(), " lon range", df.lon.min(), df.lon.max())
km_lat = 0.05 * np.pi / 180 * R
print("0.05 deg = %.2f km (lat); lon at 0N %.2f, at 10N %.2f, at 20N %.2f" % (
    km_lat, km_lat, km_lat * np.cos(np.radians(10)), km_lat * np.cos(np.radians(20))))
print("depth_m range", df.depth_m.min(), df.depth_m.max(), " NaN depth:", int(df.depth_m.isna().sum()))

fl = df["flagged"].astype(str).str.lower().isin(["true", "1", "1.0"])
print("flagged share ALL cells: %d / %d = %.4f" % (fl.sum(), len(df), fl.mean()))
print("p_flag stats:", df.p_flag.describe().round(4).to_dict())
for thr in (0.5,):
    print("p_flag>=%.2f share: %.4f" % (thr, (df.p_flag >= thr).mean()))

# nearest ERA5 good cell
gi = np.flatnonzero(good.ravel())
tree = cKDTree(xyz(LA.ravel()[gi], LO.ravel()[gi]))
chord, idx = tree.query(xyz(df.lat.values, df.lon.values))
dist_km = 2 * R * np.arcsin(np.clip(chord / 2, 0, 1))
e = gi[idx]
df["era_lat"], df["era_lon"], df["era_km"] = LA.ravel()[e], LO.ravel()[e], dist_km
has = dist_km <= 20.0
for k, arr in [("P", P), ("Te_plain", Te_plain), ("Te_pw", Te_pw), ("Hs", Hs), ("Hs2", Hrms2), ("Tp_mean", Tp_mean),
               ("ratio_mean", ratio_mean)]:
    df[k] = np.where(has, arr.ravel()[e], np.nan)
df["era_flat"] = np.where(has, e, -1)
df["flag_b"] = fl.values

band = (df.depth_m >= 30) & (df.depth_m <= 100)
print("\n--- band (30<=depth<=100)")
print("band cells:", int(band.sum()))
bw = band & has
print("band with wave data (<=20 km):", int(bw.sum()))
print("band unflagged:", int((band & ~fl).sum()), " band flagged:", int((band & fl).sum()))
print("band unflagged with wave data:", int((bw & ~fl).sum()))
print("band P range: %.3f - %.3f kW/m" % (df.P[bw].min(), df.P[bw].max()))
print("band unflagged P range: %.3f - %.3f" % (df.P[bw & ~fl].min(), df.P[bw & ~fl].max()))
print("distinct ERA5 cells used by band:", df.era_flat[bw].nunique())
# alt band definitions
print("alt: 30<depth<100:", int(((df.depth_m > 30) & (df.depth_m < 100)).sum()),
      " -100<=depth<=-30:", int(((df.depth_m >= -100) & (df.depth_m <= -30)).sum()))
for km in (10, 15, 25, 30):
    print("  band with ERA5 within %d km: %d" % (km, int((band & (dist_km <= km)).sum())))
# all cells with wave data
print("all cells with wave data:", int(has.sum()))
top = df[bw].sort_values("P", ascending=False)
print("\nTop 12 band cells by P:")
print(top[["lat", "lon", "depth_m", "dist_coast_km", "P", "Te_plain", "Te_pw", "flag_b", "subbasin", "era_lat", "era_lon"]].head(12).round(3).to_string())
print("band cells with P > 13.0:", int((df.P[bw] > 13.0).sum()), " >= 12.995:", int((df.P[bw] >= 12.995).sum()))
print("unflagged band cells with P > 13.0:", int((df.P[bw & ~fl] > 13.0).sum()))
print("distinct ERA5 cells with P>13.0 in band:", df.era_flat[bw & (df.P > 13.0)].nunique())

sites = [(7.6083, 77.9583, 13.0, 7.97, 78, 67), (8.0083, 78.2583, 10.93, 7.63, 98, 46),
         (7.5583, 77.1583, 12.26, 8.05, 83, 66), (6.6083, 81.9583, 12.32, 8.39, 33, 21),
         (6.0083, 81.4083, 12.98, 8.58, 88, 24), (6.5083, 93.8083, 17.87, 9.9, 68, 28)]
print("\n--- six sites")
rows = []
for k, (sl, so, qP, qT, qd, qkm) in enumerate(sites, 1):
    dd = hav(sl, so, df.lat.values, df.lon.values)
    j = int(np.argmin(dd))
    r = df.iloc[j]
    rank_band = int((df.P[bw] > r.P).sum()) + 1 if band.iloc[j] and has[j] else None
    rows.append(dict(site=k, cell_lat=round(r.lat, 4), cell_lon=round(r.lon, 4), off_km=round(dd[j], 3),
                     depth=r.depth_m, dist_coast=round(r.dist_coast_km, 1), P=round(r.P, 3), Te_plain=round(r.Te_plain, 3),
                     Te_pw=round(r.Te_pw, 3), Hs=round(r.Hs, 3), Tp_mean=round(r.Tp_mean, 3), flagged=bool(r.flag_b),
                     p_flag=r.p_flag, in_band=bool(band.iloc[j]), era=(r.era_lat, r.era_lon), era_km=round(r.era_km, 2),
                     rank_band=rank_band, subbasin=r.subbasin, q=(qP, qT, qd, qkm)))
for r in rows:
    print(r)
df["band"] = band
df["has"] = has
df.to_parquet(od + "/cells_with_wave.parquet") if False else df.to_pickle(od + "/cells_with_wave.pkl")
np.savez(od + "/era5_fields.npz", lat=lat, lon=lon, P=P, Te_plain=Te_plain, Te_pw=Te_pw, good=good)
