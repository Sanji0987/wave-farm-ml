"""
Coastline geometry features for ERA5 wave bias correction.

The pooled bathymetry features (depth, slope, mean fetch) failed. The
diagnostic showed the largest ERA5 errors are at geometrically sheltered
sites -- Nantucket Sound -49%, Cape Cod Bay -42% -- which is a coastline
shape effect that a 4 km, 8-sector mean fetch cannot resolve.

This module computes:
    - directional fetch at 5 degree resolution, 1 km steps
    - openness: ocean fraction of a disc at several radii
    - horizon blocking: fraction of directions hitting land within R km
    - directional statistics: min/max/anisotropy of the fetch rose
    - fetch in a given wave direction (per-timestep, needs era5_mwd)

    python geometry.py            # self-test on known sheltered vs open sites
"""
import numpy as np
import xarray as xr
from scipy import ndimage

FETCH_MAX_KM = 500.0
N_SECTORS = 72                      # 5 degree resolution
STEP_KM = 1.0
OPENNESS_RADII = [5, 10, 25, 50, 100]
BLOCK_RADII = [10, 25, 50]


class Geometry:
    """Coastline geometry lookups for one regional GEBCO tile."""

    def __init__(self, path, coarsen=4):
        ds = xr.open_dataset(path)
        self.fine = ds
        c = ds.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()
        self.lat = c.lat.values
        self.lon = c.lon.values
        self.elev = c.elevation.values
        self.land = self.elev >= 0

        self.dlat_km = 111.0 * float(self.lat[1] - self.lat[0])
        self.dlon_km = (111.0 * float(self.lon[1] - self.lon[0])
                        * np.cos(np.deg2rad(self.lat.mean())))

        gy, gx = np.gradient(self.elev, self.dlat_km * 1000, self.dlon_km * 1000)
        self.slope = np.hypot(gx, gy)
        self.dist_coast = ndimage.distance_transform_edt(
            ~self.land, sampling=[abs(self.dlat_km), abs(self.dlon_km)])

        # precompute ray offsets once
        self._bearings = np.arange(N_SECTORS) * (360.0 / N_SECTORS)
        th = np.deg2rad(self._bearings)
        nsteps = int(FETCH_MAX_KM / STEP_KM)
        k = np.arange(1, nsteps + 1)[None, :]
        self._di = (np.cos(th)[:, None] * STEP_KM / self.dlat_km * k).round().astype(int)
        self._dj = (np.sin(th)[:, None] * STEP_KM / self.dlon_km * k).round().astype(int)

    # ------------------------------------------------------------------ utils
    def _idx(self, lat, lon):
        return (int(np.abs(self.lat - lat).argmin()),
                int(np.abs(self.lon - lon).argmin()))

    def depth(self, lat, lon):
        v = self.fine.elevation.sel(lat=lat, lon=lon, method="nearest").values
        return float(-v)

    # --------------------------------------------------------- directional rose
    def fetch_rose(self, lat, lon):
        """Open-water distance (km) in N_SECTORS directions, clockwise from N."""
        i, j = self._idx(lat, lon)
        ii = i + self._di
        jj = j + self._dj
        inside = ((ii >= 0) & (ii < len(self.lat))
                  & (jj >= 0) & (jj < len(self.lon)))
        ii = np.clip(ii, 0, len(self.lat) - 1)
        jj = np.clip(jj, 0, len(self.lon) - 1)

        hit = self.land[ii, jj] | ~inside          # land, or off the tile edge
        first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, hit.shape[1])
        return first * STEP_KM

    # ------------------------------------------------------------- area metrics
    def openness(self, lat, lon):
        """Ocean fraction of a disc of radius R, for several R."""
        i, j = self._idx(lat, lon)
        out = {}
        for R in OPENNESS_RADII:
            ri = max(1, int(R / abs(self.dlat_km)))
            rj = max(1, int(R / abs(self.dlon_km)))
            i0, i1 = max(0, i - ri), min(len(self.lat), i + ri + 1)
            j0, j1 = max(0, j - rj), min(len(self.lon), j + rj + 1)
            patch = self.land[i0:i1, j0:j1]
            yy, xx = np.ogrid[i0 - i:i1 - i, j0 - j:j1 - j]
            disc = ((yy * self.dlat_km) ** 2 + (xx * self.dlon_km) ** 2) <= R ** 2
            n = disc.sum()
            out[f"open_{R}km"] = float((~patch & disc).sum() / n) if n else np.nan
        return out

    # ------------------------------------------------------------------ bundle
    def features(self, lat, lon):
        rose = self.fetch_rose(lat, lon)
        i, j = self._idx(lat, lon)

        f = {
            "depth_m": self.depth(lat, lon),
            "slope": float(self.slope[i, j]),
            "dist_coast_km": float(self.dist_coast[i, j]),
            "fetch_mean": float(rose.mean()),
            "fetch_min": float(rose.min()),
            "fetch_max": float(rose.max()),
            "fetch_median": float(np.median(rose)),
            "fetch_p25": float(np.percentile(rose, 25)),
            # anisotropy: how lopsided is the rose
            "fetch_aniso": float(rose.std() / (rose.mean() + 1e-9)),
        }
        for R in BLOCK_RADII:
            f[f"blocked_{R}km"] = float((rose < R).mean())
        f.update(self.openness(lat, lon))

        # coarse rose kept as features (12 sectors of 30 degrees)
        n_coarse = 12
        g = rose.reshape(n_coarse, N_SECTORS // n_coarse).mean(axis=1)
        for s in range(n_coarse):
            f[f"rose_{s * 30:03d}"] = float(g[s])

        f["_rose"] = rose                     # kept for per-timestep lookup
        return f

    # ------------------------------------------------- direction-aware (per row)
    @staticmethod
    def fetch_toward(rose, wave_dir_deg, halfwidth_deg=45.0):
        """Cosine-weighted fetch in the direction waves are coming FROM.

        rose is the array returned by fetch_rose (index 0 = north, clockwise).
        wave_dir_deg is ERA5 mean wave direction (degrees, coming from).
        """
        wave_dir_deg = np.asarray(wave_dir_deg, dtype=float)
        bearings = np.arange(N_SECTORS) * (360.0 / N_SECTORS)
        d = np.abs((bearings[None, :] - wave_dir_deg[:, None] + 180) % 360 - 180)
        w = np.where(d <= halfwidth_deg, np.cos(np.deg2rad(d)), 0.0)
        return (w * rose[None, :]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)


# ------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    import time

    tile = sys.argv[1] if len(sys.argv) > 1 else \
        "gebco/gebco_2026_n55.0_s25.0_w-85.0_e-60.0.nc"

    t = time.time()
    g = Geometry(tile)
    print(f"built in {time.time() - t:.1f}s\n")

    sites = [
        ("44020 Nantucket Sound  SHELTERED", 41.497, -70.283),
        ("44090 Cape Cod Bay     SHELTERED", 41.840, -70.329),
        ("44008 Nantucket 54NM   OPEN",      40.500, -69.254),
        ("44011 Georges Bank     OPEN",      41.088, -66.546),
        ("41002 South Hatteras   OPEN",      31.743, -74.955),
        ("44056 Duck FRF         NEARSHORE", 36.200, -75.714),
    ]
    print(f"{'site':36s} {'fetch_min':>9s} {'aniso':>7s} "
          f"{'blk25':>6s} {'open25':>7s} {'open50':>7s}")
    for name, la, lo in sites:
        t = time.time()
        f = g.features(la, lo)
        print(f"{name:36s} {f['fetch_min']:9.0f} {f['fetch_aniso']:7.2f} "
              f"{f['blocked_25km']:6.2f} {f['open_25km']:7.2f} "
              f"{f['open_50km']:7.2f}   ({time.time()-t:.2f}s)")
