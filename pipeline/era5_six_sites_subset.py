# Cut a small ERA5 file around the six EC3 sites (every 0.25-deg cell within 2 cells of each site).
# usage: python3 -I era5_six_sites_subset.py <era5_wave_srilanka.nc> <out.nc>
import sys
import numpy as np, xarray as xr
SRC, DST = sys.argv[1:3]
ds = xr.open_dataset(SRC)
sites = {'site1_gulf_of_mannar': (7.6083, 77.9583), 'site2_tamil_nadu': (8.0083, 78.2583), 'site3_laccadive': (7.5583, 77.1583),
         'site4_se_sri_lanka': (6.6083, 81.9583), 'site5_s_sri_lanka': (6.0083, 81.4083), 'site6_nicobar': (6.5083, 93.8083)}
lat, lon = ds.latitude.values, ds.longitude.values
li, lj = set(), set()
for la, lo in sites.values():
    i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
    li.update(range(max(0, i - 2), min(len(lat), i + 3))); lj.update(range(max(0, j - 2), min(len(lon), j + 3)))
sub = ds[['swh', 'mwp', 'pp1d', 'mwd']].isel(latitude=sorted(li), longitude=sorted(lj)).load()
sub.to_netcdf(DST, encoding={v: {'zlib': True, 'complevel': 5} for v in ['swh', 'mwp', 'pp1d', 'mwd']})
c = sub.sel(latitude=7.5, longitude=78.0)
print(dict(sub.sizes), 'site 1 wave power', round(float((0.49 * c.swh ** 2 * c.mwp).mean()), 2), 'kW/m')
