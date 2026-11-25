# Wave energy farms: where to build them and how to lay them out

Course project (data science / machine learning), BITS Pilani Dubai, 2026. Team: Surya, Rudra, Ibaad, Sanjith.

Two questions about a farm of CETO-like submerged buoys (5 m sphere, three tethers):

1. **Where to build?** ERA5 waves around southern India and Sri Lanka, 30–100 m deep. Wave power 0.49·Hs²·Te, the buoy's power curve from a wave-farm simulator, and Model 1, a screen for where ERA5 can't be trusted (logistic regression / random forest; it matches a distance-to-coast rule, it doesn't beat it).
2. **How to lay out 49 buoys?** The UCI *Large-scale Wave Energy Farm* dataset of simulated layouts. A per-buoy model (40 neighbourhood features in the wave's frame → 300 gradient-boosted trees) predicts each buoy's power; near-identical "twin" layouts are kept on one side of the train/test split (GroupKFold) so scores are fair.

Headline results and every number: [`docs/results.md`](docs/results.md). Sources: [`docs/references.md`](docs/references.md).

## What's in this repo
| Path | What it is |
| --- | --- |
| `pipeline/uci_geometry.py` | Exact repeats, wave frame, front/middle/back power, near-twin pairs and families, leak share |
| `pipeline/train_perbuoy.py` | 40 per-buoy features, 5-fold GroupKFold (by layout, by family), final model, tree export |
| `pipeline/era5_sites.py` | ERA5 wave power per cell, joined to coastal cells 30–100 m deep, GEBCO land mask |
| `pipeline/era5_six_sites_subset.py` | Small ERA5 file around the six recommended sites |
| `analysis/model_comparison.py` | Linear vs quadratic vs boosted trees on the same fair split |
| `analysis/deck_figures.py` | Figures used in the presentation |
| `demo/wave_farm_demo.ipynb` | Live demo notebook (runs in under a minute on a 2,000-layout subsample) |
| `demo/Wave_Farm_Lab.html` | Interactive offline demo: drag buoys, the model predicts power live in the browser |
| `model1/` | Model 1 (where ERA5 can't be trusted): `pick_stations.py` (NOAA buoy stations), `era5_download.py` (ERA5 via the CDS API), `geometry.py` (coastline/depth features from GEBCO: distance to coast, slope, fetch rose), `build_pairs.py` (buoy + ERA5 + geometry → `pairs.csv`), `shelter_classifier.py` (logistic regression and random forest, leave-one-region-out, AUC: the original 28-feature models), `model1_five_features.py` (re-check on the 5 features in `station_summary.csv`, plus the distance-to-coast baseline) |
| `simulator/` | Our scripts around the third-party simulator: `climate.m` (sea-state tables as in the simulator's `Main.m`), `run_orig.m` (simulate any layout with a given PTO stiffness/damping and interaction cut-off kR), `kR_cutoff.patch` (one-line change making kR settable), `best_rows.py`, `ds_iso.py` (pull layouts and the lone-buoy power out of the dataset) |
| `transfer/` | Transfer tests for the per-buoy model (`transfer.py`, `common.py`: Perth 49 → 100, Sydney 49 → 100, Perth ↔ Sydney); `wec_audit_early.py` is the first version (duplicates, leak and transfer on simpler features) |
| `checks/dataset/` | Independent re-check of the dataset numbers (rows, repeats, thirds, twins) |
| `checks/published_pipeline/` | Re-implementation of the published pipeline and the raw-coordinate / shape-feature models under random, exact-grouped and family-grouped splits (`prep.py`, `cv.py`; needs LightGBM) |
| `checks/era5_sites/` | Independent re-check of the ERA5 and site numbers (band cells, six sites, periods, independent-site count, GEBCO depths) |
| `demo/src/` | How the demo is built: `wfm_core.js` (features + tree walk in JavaScript, identical to Python), `export_demo_data.py`, `build_demo.py`, `app.html`, `verify_js.js`, `optimise_layout.js`, `test_demo.js` |

## Not in this repo
- **Data** (download it yourself, see [`data/README.md`](data/README.md)): UCI layouts, ERA5, GEBCO, the coastal-cell table.
- **The wave-farm simulator**: third-party MATLAB code (Sergiienko, MATLAB Central File Exchange #71840, BSD licence), run in GNU Octave.
- **Code lost in a workspace reset in October 2026 and not yet rebuilt**: the closed optimisation loop with its controls, the simulator calibration search, the Gaussian-process comparison and the site-energy calculation. Their results are recorded in `docs/results.md`, tagged as not re-checked. The `checks/` scripts were written in the 13 Nov 2026 re-check to reproduce the other lost analyses.

## Run it
```bash
pip install -r requirements.txt
# question 2: layouts (put WEC_Perth_49.csv from the UCI zip in data/)
python3 -I pipeline/uci_geometry.py data/WEC_Perth_49.csv out
python3 -I pipeline/train_perbuoy.py out cv          # ~10-20 min on a laptop
python3 -I analysis/model_comparison.py out
# question 1: waves (ERA5 NetCDF, coastal cells, GEBCO tile in data/)
python3 -I pipeline/era5_sites.py data/era5_wave_srilanka.nc data/msr_cells.csv data/gebco_2026_n20.0_s-5.0_w70.0_e95.0.nc out
# Model 1 (needs internet for NOAA + a CDS account for ERA5; GEBCO tile in gebco/)
# order: pick_stations.py -> era5_download.py -> build_pairs.py -> shelter_classifier.py (see each script's docstring for arguments)
# transfer tests (all four UCI csv files in data/)
python3 transfer/transfer.py data out/transfer.json
# simulator (GNU Octave + octave-netcdf; simulator unzipped to sim/, kR_cutoff.patch applied)
octave simulator/climate.m sim out sim          # args: simulator dir, output dir, dir with the HCtimeseries .nc files
octave simulator/run_orig.m sim out/climate_Perth.mat layout.csv 407400 96500 200 result.txt
# demo
jupyter notebook demo/wave_farm_demo.ipynb           # or open demo/Wave_Farm_Lab.html in a browser
```
