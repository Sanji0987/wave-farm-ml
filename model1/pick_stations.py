"""
Find NDBC stations inside the project regions that have usable data.

    python pick_stations.py

Writes stations.csv with one row per usable station. Does not download the
data itself -- this is only the shortlist.
"""
import io
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

YEARS = [2016, 2017]

# name: [North, West, South, East]  -- must match era5_download.py
REGIONS = {
    "westcoast": [55, -135, 25, -110],
    "eastcoast": [55, -85, 25, -60],
    "gulf":      [33, -100, 15, -78],
    "hawaii":    [28, -167, 13, -147],
    "srilanka":  [20, 70, -5, 95],
}

STATIONS_XML = "https://www.ndbc.noaa.gov/activestations.xml"
HIST = "https://www.ndbc.noaa.gov/data/historical/stdmet/{stn}h{year}.txt.gz"


def region_of(lat, lon):
    for name, (n, w, s, e) in REGIONS.items():
        if s <= lat <= n and w <= lon <= e:
            return name
    return None


def fetch_station_list():
    print("fetching station list ...")
    with urllib.request.urlopen(STATIONS_XML, timeout=60) as r:
        root = ET.parse(io.BytesIO(r.read())).getroot()

    rows = []
    for st in root.findall("station"):
        try:
            lat = float(st.get("lat"))
            lon = float(st.get("lon"))
        except (TypeError, ValueError):
            continue
        reg = region_of(lat, lon)
        if reg is None:
            continue
        rows.append({
            "station": st.get("id").lower(),
            "name": st.get("name"),
            "lat": lat,
            "lon": lon,
            "type": st.get("type"),
            "owner": st.get("owner"),
            "met": st.get("met"),
            "region": reg,
        })
    return pd.DataFrame(rows)


def has_year(stn, year):
    """HEAD request -- does the historical file exist?"""
    url = HIST.format(stn=stn, year=year)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    df = fetch_station_list()
    print(f"{len(df)} stations inside the regions")
    df = df[df["met"] == "y"]
    print(f"{len(df)} report meteorological data")

    print(f"\nchecking data availability for {YEARS} "
          f"({len(df)} stations, this takes a few minutes) ...")
    keep = []
    for i, row in enumerate(df.itertuples(), 1):
        years = [y for y in YEARS if has_year(row.station, y)]
        if len(years) == len(YEARS):
            keep.append(row.station)
            mark = "ok"
        else:
            mark = f"only {years}" if years else "none"
        print(f"  [{i}/{len(df)}] {row.station} {row.region:10s} {mark}")

    out = df[df["station"].isin(keep)].sort_values(["region", "station"])
    out.to_csv("stations.csv", index=False)

    print(f"\n{len(out)} stations have full coverage for {YEARS}")
    print(out.groupby("region").size().to_string())
    print("\nwritten to stations.csv")
    if len(out) < 15:
        print("\nWARNING: fewer than 15 stations. Consider widening the regions,"
              " changing YEARS, or adding CDIP stations.")


if __name__ == "__main__":
    sys.exit(main())
