"""
ERA5 wave data download, chunked by month to stay under the CDS cost limit.

    python era5_download.py            # download monthly chunks
    python era5_download.py --merge    # concatenate into one file per region

Safe to re-run: existing files are skipped. Kill and restart freely.
"""
import argparse
import glob
import os
import sys
import time
import traceback

OUT = "era5"
MERGED = "era5_merged"

# ---------------------------------------------------------------- configuration
YEARS = [2016, 2017]          # start small; add years once this works

TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "peak_wave_period",
    "mean_wave_direction",
]

# name: [North, West, South, East]
REGIONS = {
    "westcoast": [55, -135, 25, -110],
    "eastcoast": [55, -85, 25, -60],
    "gulf":      [33, -100, 15, -78],
    "hawaii":    [28, -167, 13, -147],
    "srilanka":  [20, 70, -5, 95],
}


def path_for(region, year, month):
    return os.path.join(OUT, f"era5_wave_{region}_{year}{month:02d}.nc")


# --------------------------------------------------------------------- download
def request(client, region, area, year, month):
    target = path_for(region, year, month)
    if os.path.exists(target) and os.path.getsize(target) > 10_000:
        return "skip"

    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": VARIABLES,
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": TIMES,
                "area": area,
                "grid": [0.25, 0.25],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            target,
        )
        return "ok"
    except Exception:
        traceback.print_exc(limit=1)
        if os.path.exists(target):
            os.remove(target)
        return "fail"


def download():
    import cdsapi
    os.makedirs(OUT, exist_ok=True)
    client = cdsapi.Client()

    jobs = [(r, a, y, m)
            for y in YEARS
            for m in range(1, 13)
            for r, a in REGIONS.items()]
    print(f"{len(jobs)} monthly chunks queued\n")

    failures = []
    for i, (region, area, year, month) in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {region} {year}-{month:02d}"
        status = request(client, region, area, year, month)
        if status == "skip":
            print(f"{tag}  skip")
        elif status == "ok":
            mb = os.path.getsize(path_for(region, year, month)) / 1e6
            print(f"{tag}  ok  ({mb:.0f} MB)", flush=True)
        else:
            print(f"{tag}  FAILED", flush=True)
            failures.append(tag)
        time.sleep(1)

    print("\n" + "=" * 50)
    if failures:
        print(f"{len(failures)} failed. Re-run to retry:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all chunks downloaded - now run:  python era5_download.py --merge")


# ------------------------------------------------------------------------ merge
def merge():
    import xarray as xr
    os.makedirs(MERGED, exist_ok=True)
    for region in REGIONS:
        files = sorted(glob.glob(os.path.join(OUT, f"era5_wave_{region}_*.nc")))
        if not files:
            print(f"{region}: no chunks found, skipping")
            continue
        out = os.path.join(MERGED, f"era5_wave_{region}.nc")
        print(f"{region}: merging {len(files)} chunks ...", flush=True)
        parts = [xr.open_dataset(f, engine="netcdf4").load() for f in files]
        ds = xr.concat(parts, dim="valid_time").sortby("valid_time")
        ds.to_netcdf(out)
        ds.close()
        for p in parts:
            p.close()
        print(f"  -> {out}  ({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true",
                   help="concatenate downloaded monthly chunks per region")
    args = p.parse_args()
    merge() if args.merge else download()
