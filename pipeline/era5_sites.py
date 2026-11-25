# Corrected wave power per ERA5 cell (Te = mwp), joined to coastal cells; land mask from GEBCO.
# usage: python3 -I era5_sites.py <era5_srilanka.nc> <msr_cells.csv> <gebco.nc> <out_dir>
import sys, json
import numpy as np, pandas as pd, xarray as xr

NC, CELLS, GEB, OUT = sys.argv[1:5]
ds = xr.open_dataset(NC)
nt = ds.sizes['valid_time']
lat = ds.latitude.values; lon = ds.longitude.values
acc = np.zeros((len(lat), len(lon))); n = np.zeros_like(acc)
for a in range(0, nt, 400):
    sl = slice(a, min(nt, a + 400))
    hs = ds.swh.isel(valid_time=sl).values.astype(float)
    te = ds.mwp.isel(valid_time=sl).values.astype(float)
    w = 0.49 * hs ** 2 * te
    ok = np.isfinite(w)
    acc += np.where(ok, w, 0).sum(0); n += ok.sum(0)
wpd = np.where(n > 0.9 * nt, acc / np.maximum(n, 1), np.nan)
print('ERA5 cells with data', int(np.isfinite(wpd).sum()))

def cell(la, lo):
    i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
    return lat[i], lon[j], wpd[i, j]
for name, la, lo in [('top site', 7.6083, 77.9583), ('old #1 by wave power', 6.5083, 93.8083)]:
    print(name, 'nearest ERA5 cell', cell(la, lo))

c = pd.read_csv(CELLS, usecols=['lat', 'lon', 'depth_m', 'dist_coast_km', 'p_flag', 'flagged', 'subbasin'])
# nearest ERA5 cell with data, within 20 km
LA, LO = np.meshgrid(lat, lon, indexing='ij')
valid = np.isfinite(wpd)
vl, vo, vw = LA[valid], LO[valid], wpd[valid]
from scipy.spatial import cKDTree
kx = lambda la, lo: np.c_[la * 111.0, lo * 111.0 * np.cos(np.deg2rad(7.5))]
tree = cKDTree(kx(vl, vo))
d, k = tree.query(kx(c.lat.values, c.lon.values))
c['wpd'] = np.where(d <= 20, vw[k], np.nan)
c['era5_km'] = d
band = c[(c.depth_m >= 30) & (c.depth_m <= 100)].copy()
print('cells', len(c), 'in 30-100 m band', len(band), 'band & not flagged', int((~band.flagged).sum()),
      'band & resource matched', int(band.wpd.notna().sum()))
print('band wpd range', np.nanmin(band.wpd).round(1), np.nanmax(band.wpd).round(1))
band[['lat', 'lon', 'depth_m', 'dist_coast_km', 'p_flag', 'flagged', 'subbasin', 'wpd']].to_csv(f'{OUT}/band_cells.csv', index=False)

# land mask from GEBCO, downsampled
g = xr.open_dataset(GEB)
el = g.elevation.values[::8, ::8]
glat = g.lat.values[::8]; glon = g.lon.values[::8]
np.savez_compressed(f'{OUT}/landmask.npz', land=(el > 0), lat=glat, lon=glon)
np.savez_compressed(f'{OUT}/era5_wpd.npz', wpd=wpd, lat=lat, lon=lon)
print('saved', el.shape)
