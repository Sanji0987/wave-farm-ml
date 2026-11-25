"""
Build the paired dataset: buoy observations + ERA5 reanalysis + GEBCO features.

    python build_pairs.py

Writes pairs.csv -- one row per station per 3-hourly timestep, ready for the
evaluation harness (features, target, groups).

Expects:
    era5_merged/era5_wave_<region>.nc      from era5_download.py --merge
    gebco/gebco_2026_*.nc                  from download.gebco.net
    ndbc/                                  created here, cached downloads
"""
import gzip
import io
import os
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr
from geometry import Geometry

# --------------------------------------------------------------------- config
# Set PILOT = None to use every wave buoy in stations.csv (recommended once
# the pilot run has confirmed the pipeline works).
PILOT = None

# stations rejected after bathymetry lookup -- estuarine / fetch-limited sites
MIN_DEPTH_M = 15.0
MIN_FETCH_KM = 40.0
MIN_RECORDS = 1500

YEARS = [2016, 2017]
GEBCO_DIR = "gebco"
ERA5_DIR = "era5_merged"
NDBC_DIR = "ndbc"

RHO, G = 1025.0, 9.81
K_WPD = RHO * G**2 / (64 * np.pi) / 1000.0     # -> kW/m

FETCH_MAX_KM = 500.0
FETCH_SECTORS = 8

# ------------------------------------------------------------------ ndbc load
NDBC_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{s}h{y}.txt.gz"


def load_ndbc(station, year):
    """Return a DataFrame with time, WVHT, DPD, APD, MWD (missing -> NaN)."""
    os.makedirs(NDBC_DIR, exist_ok=True)
    cache = os.path.join(NDBC_DIR, f"{station}h{year}.txt")

    if not os.path.exists(cache):
        url = NDBC_URL.format(s=station, y=year)
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = gzip.decompress(r.read()).decode("utf-8", "replace")
        with open(cache, "w") as f:
            f.write(raw)
    else:
        raw = open(cache).read()

    lines = raw.splitlines()
    header = lines[0].lstrip("#").split()
    # second line is units in modern files; detect and skip
    skip = 2 if lines[1].startswith("#") else 1
    df = pd.read_csv(io.StringIO("\n".join(lines[skip:])),
                     sep=r"\s+", names=header, na_values=["MM"])

    df["time"] = pd.to_datetime(
        dict(year=df["YY"], month=df["MM"], day=df["DD"],
             hour=df["hh"], minute=df.get("mm", 0)),
        errors="coerce", utc=True)

    for col, miss in [("WVHT", 99.0), ("DPD", 99.0), ("APD", 99.0), ("MWD", 999)]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] >= miss, col] = np.nan
        else:
            df[col] = np.nan

    return df[["time", "WVHT", "DPD", "APD", "MWD"]].dropna(subset=["time"])


def buoy_series(station):
    parts = []
    for y in YEARS:
        try:
            parts.append(load_ndbc(station, y))
        except Exception as e:
            print(f"    {station} {y}: {e}")
    if not parts:
        return None
    df = pd.concat(parts).sort_values("time")
    # collapse to the 3-hourly ERA5 grid
    df = df.set_index("time").resample("3h").mean().dropna(subset=["WVHT"])
    return df.reset_index()


# ------------------------------------------------------------- gebco features
def find_gebco(region_box):
    """Pick the GEBCO tile matching a region by filename bounds."""
    n, w, s, e = region_box
    for fn in os.listdir(GEBCO_DIR):
        if fn.endswith(".nc") and f"n{float(n)}" in fn and f"w{float(w)}" in fn:
            return os.path.join(GEBCO_DIR, fn)
    raise FileNotFoundError(f"no GEBCO tile for {region_box}")


# ---------------------------------------------------------------- era5 lookup
def era5_at(ds, lat, lon):
    """Nearest ocean grid cell time series as a DataFrame."""
    sub = ds.sel(latitude=lat, longitude=lon, method="nearest")
    if np.isnan(sub.swh.values).all():           # landed on a land cell
        la = ds.latitude.values
        lo = ds.longitude.values
        # a cell is ocean if swh is finite at any time in a short sample
        ocean = np.isfinite(ds.swh.isel(valid_time=slice(0, 8)).values).any(0)
        if not ocean.any():
            raise ValueError("no ocean cells in tile")
        d2 = ((la[:, None] - lat) ** 2) + ((lo[None, :] - lon) ** 2)
        d2 = np.where(ocean, d2, np.inf)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        sub = ds.isel(latitude=int(i), longitude=int(j))

    return pd.DataFrame({
        "time": pd.to_datetime(sub.valid_time.values, utc=True),
        "era5_swh": sub.swh.values,
        "era5_mwp": sub.mwp.values,
        "era5_pp1d": sub.pp1d.values,
        "era5_mwd": sub.mwd.values,
    })


# ----------------------------------------------------------------------- wpd
def wpd(hs, te):
    return K_WPD * hs**2 * te


# ---------------------------------------------------------------------- main
REGION_BOX = {
    "westcoast": [55, -135, 25, -110],
    "eastcoast": [55, -85, 25, -60],
    "gulf":      [33, -100, 15, -78],
    "hawaii":    [28, -167, 13, -147],
}


def main():
    stations = pd.read_csv("stations.csv", dtype={"station": str})

    if PILOT is None:
        sel = stations[stations["type"] == "buoy"].copy()
        sel = sel[~sel["station"].str.startswith("45")]      # Great Lakes
        sel = sel[sel["region"].isin(REGION_BOX)]
        targets = list(zip(sel["station"], sel["region"]))
        print(f"{len(targets)} candidate wave buoys\n")
    else:
        targets = list(PILOT.items())

    stations = stations.set_index("station")
    bathy_cache, era5_cache = {}, {}
    rows, rejected = [], []

    for stn, region in targets:
        if stn not in stations.index:
            continue
        lat = float(stations.loc[stn, "lat"])
        lon = float(stations.loc[stn, "lon"])
        buoy = buoy_series(stn)
        if buoy is None or len(buoy) < MIN_RECORDS:
            n = 0 if buoy is None else len(buoy)
            rejected.append((stn, f"only {n} records"))
            continue

        if region not in era5_cache:
            era5_cache[region] = xr.open_dataset(
                os.path.join(ERA5_DIR, f"era5_wave_{region}.nc"))
        if region not in bathy_cache:
            bathy_cache[region] = Geometry(find_gebco(REGION_BOX[region]))

        feats = bathy_cache[region].features(lat, lon)
        if feats["depth_m"] < MIN_DEPTH_M:
            rejected.append((stn, f"depth {feats['depth_m']:.0f} m"))
            continue
        if feats["fetch_max"] < MIN_FETCH_KM:
            rejected.append((stn, f"max fetch {feats['fetch_max']:.0f} km"))
            continue

        era5 = era5_at(era5_cache[region], lat, lon)
        df = buoy.merge(era5, on="time", how="inner")
        if df.empty:
            rejected.append((stn, "no overlapping timestamps"))
            continue

        # two Te conventions -- report both, decide later
        # Te from peak period where available, else fall back to average
        # period (many CDIP stations report APD but not DPD)
        te_buoy = (0.90 * df.DPD).fillna(1.10 * df.APD)
        df["te_source"] = np.where(df.DPD.notna(), "tp", "tz")
        df["buoy_wpd_tp"] = wpd(df.WVHT, te_buoy)
        df["buoy_wpd_tz"] = wpd(df.WVHT, 1.10 * df.APD)
        df["era5_wpd_tp"] = wpd(df.era5_swh, 0.90 * df.era5_pp1d)
        df["era5_wpd_tz"] = wpd(df.era5_swh, 1.10 * df.era5_mwp)

        df["resid_hs"] = df.WVHT - df.era5_swh
        df["resid_wpd_tp"] = df.buoy_wpd_tp - df.era5_wpd_tp
        df["resid_wpd_tz"] = df.buoy_wpd_tz - df.era5_wpd_tz

        rose = feats.pop("_rose")
        for k, v in feats.items():
            df[k] = v
        # fetch in the direction the waves are arriving from (per timestep)
        df["fetch_toward"] = Geometry.fetch_toward(rose, df.era5_mwd.values)
        df["station"] = stn
        df["region"] = region
        df["lat"] = lat
        df["lon"] = lon
        df["month"] = df.time.dt.month

        rows.append(df)
        print(f"{stn} {region:10s} depth {feats['depth_m']:6.0f} m  "
              f"coast {feats['dist_coast_km']:5.1f} km  {len(df):5d} rows",
              flush=True)

    if not rows:
        print("\nnothing built")
        return

    out = pd.concat(rows, ignore_index=True)
    out.to_csv("pairs.csv", index=False)

    print(f"\n{len(out)} rows, {out.station.nunique()} stations -> pairs.csv")
    print(out.groupby("region").station.nunique().to_string())
    if rejected:
        print(f"\n{len(rejected)} rejected:")
        for s, why in rejected:
            print(f"  {s}  {why}")

    # headline check: does per-station bias track depth?
    per = (out.groupby("station")
              .agg(bias=("resid_hs", "mean"), sd=("resid_hs", "std"),
                   depth=("depth_m", "first"), coast=("dist_coast_km", "first"))
              .dropna())
    if len(per) > 3:
        ld = np.log10(per.depth)
        print(f"\ncorr(station bias, log depth) = {per.bias.corr(ld):+.3f}"
              f"   spearman {per.bias.corr(per.depth, method='spearman'):+.3f}"
              f"   (n={len(per)})")
        print(f"bias range {per.bias.max()-per.bias.min():.3f} m,"
              f" mean within-station sd {per.sd.mean():.3f} m")


if __name__ == "__main__":
    main()
