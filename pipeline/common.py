# Shared helpers for the rebuilt analyses: load any UCI file, unique layouts, per-buoy features in the wave frame.
# The feature function is a vectorised version of pipeline/train_perbuoy.py::feats (checked identical in check_feats()).
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

K = 12; RADII = [150, 300, 600, 1200]; CAP = 2000.0
P_ISO = {'Perth': 96374.0, 'Sydney': 105788.0}   # dataset's lone-buoy power (results.md §1)
THETA_PERTH = 55.3


def load_unique(csv):
    """-> X, Y (nu, N) positions; P (nu, N) per-buoy power; T (nu,) total power. Duplicates averaged."""
    df = pd.read_csv(csv)
    N = sum(1 for c in df.columns if c.startswith('X') and c[1:].isdigit())
    X = df[[f'X{i}' for i in range(1, N + 1)]].to_numpy(float)
    Y = df[[f'Y{i}' for i in range(1, N + 1)]].to_numpy(float)
    PW = df[[f'Power{i}' for i in range(1, N + 1)]].to_numpy(float)
    TP = df['Total_Power'].to_numpy(float)
    coords = np.round(np.concatenate([X, Y], 1), 6)
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    inv = inv.ravel()
    P = pd.DataFrame(PW).groupby(inv).mean().sort_index().to_numpy()
    T = pd.Series(TP).groupby(inv).mean().sort_index().to_numpy()
    return uniq[:, :N], uniq[:, N:], P, T


def feats(x, y, theta_deg=THETA_PERTH):
    """x, y: (n,) -> (n, 40). Same features, same order as the demo / train_perbuoy.py."""
    th = np.deg2rad(theta_deg); C, S = np.cos(th), np.sin(th)
    u = x * C + y * S; v = -x * S + y * C
    n = len(x)
    du = u[None, :] - u[:, None]; dv = v[None, :] - v[:, None]
    dd = np.sqrt(du * du + dv * dv)
    eye = np.eye(n, dtype=bool)
    dd_s = np.where(eye, np.inf, dd)
    o = np.argsort(dd_s, axis=1, kind='stable')[:, :K]
    F = np.zeros((n, 40))
    F[:, 0:2 * K:2] = np.take_along_axis(du, o, 1)
    F[:, 1:2 * K:2] = np.take_along_axis(dv, o, 1)
    a = np.abs(np.degrees(np.arctan2(dv, du)))
    ne = ~eye
    up = (a <= 30) & ne; dn = (a >= 150) & ne; sd = (a >= 60) & (a <= 120) & ne
    col = 2 * K
    for R in RADII: F[:, col] = np.sum(up & (dd <= R), 1); col += 1
    for R in RADII: F[:, col] = np.sum(dn & (dd <= R), 1); col += 1
    for m in (up, dn, sd):
        F[:, col] = np.minimum(CAP, np.where(m, dd, np.inf).min(1)); col += 1
    F[:, col] = np.sum((du > 0) & ne, 1) / (n - 1); col += 1
    F[:, col] = u - u.mean(); F[:, col + 1] = v - v.mean(); col += 2
    F[:, col] = np.sum((dd <= 150) & ne, 1); F[:, col + 1] = np.sum((dd <= 300) & ne, 1)
    return F


def feats_all(X, Y, theta_deg=THETA_PERTH):
    return np.concatenate([feats(X[k], Y[k], theta_deg) for k in range(len(X))])


def model(it=300, seed=0):
    return HistGradientBoostingRegressor(max_iter=it, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, early_stopping=False, random_state=seed)


def farm_metrics(t, p):
    """t, p: true and predicted farm power (W). Returns R2, debiased R2, rho, top-5% hit rate, mean bias (kW)."""
    res = t - p
    sst = np.sum((t - t.mean()) ** 2)
    r2 = 1 - np.sum(res ** 2) / sst
    r2d = 1 - np.sum((res - res.mean()) ** 2) / sst
    k = max(1, len(t) // 20)
    top_t = set(np.argsort(-t)[:k]); top_p = set(np.argsort(-p)[:k])
    return {'r2': round(float(r2), 3), 'r2_debiased': round(float(r2d), 3), 'rho': round(float(spearmanr(t, p)[0]), 3),
            'top5_found': round(len(top_t & top_p) / k, 3), 'bias_kW': round(float(-res.mean()) / 1e3, 1), 'n': int(len(t))}


def fit_theta(X, Y, P, grid=np.arange(0, 360, 0.5)):
    """Wave direction that maximises mean within-layout corr(power, up-wave coordinate)."""
    best = None
    sub = np.random.default_rng(0).choice(len(X), min(2000, len(X)), replace=False)
    Xs, Ys, Ps = X[sub], Y[sub], P[sub]
    Pc = Ps - Ps.mean(1, keepdims=True); Pn = np.sqrt((Pc ** 2).sum(1))
    for t in grid:
        th = np.deg2rad(t); U = Xs * np.cos(th) + Ys * np.sin(th)
        Uc = U - U.mean(1, keepdims=True)
        c = np.nanmean((Uc * Pc).sum(1) / (np.sqrt((Uc ** 2).sum(1)) * Pn + 1e-12))
        if best is None or c > best[1]: best = (t, c)
    return best


def check_feats():
    import importlib.util, sys
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 1000, 49), rng.uniform(0, 1000, 49)
    # reference: loop version copied from pipeline/train_perbuoy.py
    TH = np.deg2rad(55.3); C, S = np.cos(TH), np.sin(TH)
    u = x * C + y * S; v = -x * S + y * C
    du = u[None, :] - u[:, None]; dv = v[None, :] - v[:, None]; dd = np.sqrt(du * du + dv * dv)
    n = len(x); F = np.zeros((n, 40)); um, vm = u.mean(), v.mean()
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i]); di, ui, vi = dd[i, idx], du[i, idx], dv[i, idx]
        o = np.argsort(di, kind='stable')[:K]; F[i, 0:2 * K:2] = ui[o]; F[i, 1:2 * K:2] = vi[o]
        a = np.abs(np.degrees(np.arctan2(vi, ui))); up, dn, sd = a <= 30, a >= 150, (a >= 60) & (a <= 120); col = 2 * K
        for R in RADII: F[i, col] = np.sum(up & (di <= R)); col += 1
        for R in RADII: F[i, col] = np.sum(dn & (di <= R)); col += 1
        for msk in (up, dn, sd): F[i, col] = min(CAP, di[msk].min()) if msk.any() else CAP; col += 1
        F[i, col] = np.sum(ui > 0) / (n - 1); col += 1
        F[i, col] = u[i] - um; F[i, col + 1] = v[i] - vm; col += 2
        F[i, col] = np.sum(di <= 150); F[i, col + 1] = np.sum(di <= 300)
    return float(np.abs(F - feats(x, y)).max())


if __name__ == '__main__':
    print('max abs diff vs loop version:', check_feats())
