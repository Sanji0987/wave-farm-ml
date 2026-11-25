# Data (not committed)

| File | Source | Used by |
| --- | --- | --- |
| `WEC_Perth_49.csv` (+ Sydney 49, Perth 100, Sydney 100) | UCI Machine Learning Repository, "Large-scale Wave Energy Farm" (Neshat et al.) | `pipeline/uci_geometry.py`, `train_perbuoy.py`, the demo notebook (it also accepts the UCI zip itself) |
| `era5_wave_srilanka.nc` | Copernicus Climate Data Store, ERA5 hourly single levels: `swh`, `mwp`, `pp1d`, `mwd`, 3-hourly 2016-01-01 to 2017-12-31, 5°S–20°N, 70–95°E, 0.25° | `pipeline/era5_sites.py` |
| `msr_cells.csv` | the team's coastal-cell table (41,396 cells: depth, distance to coast, Model 1 flags) | `era5_sites.py`, the demo notebook's Model 1 screen |
| `gebco_2026_n20.0_s-5.0_w70.0_e95.0.nc` | GEBCO gridded bathymetry, subset download (gebco.net) | `era5_sites.py` (land mask) |
| `station_summary.csv` | the team's table of 94 US buoy stations (ERA5 vs buoy wave-height bias) | Model 1; also embedded in the demo notebook |
