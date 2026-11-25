# Geometry stats + near-twin families for the UCI Perth 49 file.
# usage: python3 -I uci_geometry.py <WEC_Perth_49.csv> <out_dir>
import sys, json
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CSV, OUT = sys.argv[1], sys.argv[2]
N = 49
P_ISO = 96374.0           # dataset's own isolated-buoy power, Perth (W)
THETA = np.deg2rad(55.3)  # waves arrive FROM 55.3 deg in layout coordinates

df = pd.read_csv(CSV)
X = df[[f'X{i}' for i in range(1, N + 1)]].to_numpy(float)
Y = df[[f'Y{i}' for i in range(1, N + 1)]].to_numpy(float)
PW = df[[f'Power{i}' for i in range(1, N + 1)]].to_numpy(float)
TP = df['Total_Power'].to_numpy(float)
rows = len(df)

# ---- exact duplicates: unique coordinate tuples (buoy order kept)
coords = np.round(np.concatenate([X, Y], axis=1), 6)
uniq, first, inv = np.unique(coords, axis=0, return_index=True, return_inverse=True)
inv = inv.ravel()
nu = len(uniq)
g = pd.DataFrame(PW).groupby(inv).mean().sort_index().to_numpy()
tp = pd.Series(TP).groupby(inv).mean().sort_index().to_numpy()
cnt = np.bincount(inv)
UX, UY = uniq[:, :N], uniq[:, N:]
print(f'rows {rows}, unique {nu}, duplicated {100*(1-nu/rows):.1f}%')

# ---- wave frame: u = up-wave coordinate (bigger u = nearer where waves come from)
c, s = np.cos(THETA), np.sin(THETA)
U = UX * c + UY * s
V = -UX * s + UY * c
corr = np.mean([np.corrcoef(U[k], g[k])[0, 1] for k in range(nu)])
print(f'mean within-layout corr(power, up-wave coordinate) = {corr:+.2f}')

# thirds by up-wave rank
rank = np.argsort(np.argsort(-U, axis=1), axis=1)  # 0 = most up-wave
front = g[rank < N / 3].mean(); mid = g[(rank >= N / 3) & (rank < 2 * N / 3)].mean(); back = g[rank >= 2 * N / 3].mean()
print(f'front {front/1e3:.1f} kW, middle {mid/1e3:.1f} kW, back {back/1e3:.1f} kW, back vs front {100*(back/front-1):.1f}%')

# shadowed = any neighbour within +-30 deg of the up-wave direction (any distance)
shadow = np.zeros_like(g, bool)
for k in range(nu):
    du = U[k][None, :] - U[k][:, None]; dv = V[k][None, :] - V[k][:, None]
    ang = np.degrees(np.arctan2(dv, du)); np.fill_diagonal(ang, 999)
    shadow[k] = (np.abs(ang) <= 30).any(axis=1)
print(f'no up-wave neighbour: {g[~shadow].mean()/1e3:.1f} kW (n={(~shadow).sum()}), shadowed: {g[shadow].mean()/1e3:.1f} kW')

# ---- near-twin families: layouts differing by exactly one buoy (48 positions identical)
def mix(z):
    z = (z + np.uint64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))
with np.errstate(over='ignore'):
    pk = (np.round(UX * 10).astype(np.int64) * 1_000_003 + np.round(UY * 10).astype(np.int64)).astype(np.uint64)
    h = mix(pk)
    S = h.sum(axis=1, dtype=np.uint64)
    keys = (S[:, None] - h).ravel()          # layout hash with buoy i removed
lay = np.repeat(np.arange(nu), N); buoy = np.tile(np.arange(N), nu)
order = np.argsort(keys, kind='stable')
ks, ls, bs = keys[order], lay[order], buoy[order]
# FIX (13 Nov 2026): link EVERY pair of layouts in a bucket of equal keys, not only sort-order neighbours.
# The old version linked adjacent entries only (k-1 pairs per bucket instead of k(k-1)/2): it found the
# same families but undercounted pairs (2,252 instead of 156,649) and gave a meaningless 71 m median move.
starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]); ends = np.r_[starts[1:], len(ks)]
E = []
for s0, e0 in zip(starts, ends):
    if e0 - s0 < 2: continue
    ii, jj = np.triu_indices(e0 - s0, 1); ii += s0; jj += s0
    k_ = ls[ii] != ls[jj]
    E.append(np.c_[ls[ii][k_], ls[jj][k_], bs[ii][k_], bs[jj][k_]])
ea, eb, ia, ib = np.concatenate(E).T
moved = np.hypot(UX[ea, ia] - UX[eb, ib], UY[ea, ia] - UY[eb, ib])
pairs = pd.DataFrame({'a': np.minimum(ea, eb), 'b': np.maximum(ea, eb), 'moved': moved}).drop_duplicates(['a', 'b'])
A = coo_matrix((np.ones(len(pairs)), (pairs.a, pairs.b)), shape=(nu, nu))
ncomp, fam = connected_components(A, directed=False)
sizes = np.bincount(fam)
best = int(np.argmax(tp))
print(f'one-buoy-difference pairs {len(pairs)}, moved distance median {pairs.moved.median():.0f} m')
_fb = fam[np.argmax(tp)]; _in = (fam[pairs.a.to_numpy()] == _fb)
print(f'  pairs inside the best family {_in.sum()}, median move {pairs.moved[_in].median():.0f} m; '
      f'outside it {(~_in).sum()}, median move {pairs.moved[~_in].median():.0f} m')
print(f'families {ncomp}; singletons {(sizes==1).sum()}; largest {sizes.max()} ({100*sizes.max()/nu:.1f}%); '
      f'best layout family size {sizes[fam[best]]}')
has_twin = np.zeros(nu, bool); has_twin[pairs.a] = True; has_twin[pairs.b] = True
print(f'layouts with at least one one-buoy twin: {100*has_twin.mean():.1f}%')

# share of test layouts with a one-buoy twin in training under an exact-layout 5-fold split
rng = np.random.default_rng(0)
fold = rng.integers(0, 5, nu)
cross = fold[pairs.a.to_numpy()] != fold[pairs.b.to_numpy()]
leak = np.zeros(nu, bool); leak[pairs.a.to_numpy()[cross]] = True; leak[pairs.b.to_numpy()[cross]] = True
print(f'exact-layout 5-fold split: {100*leak.mean():.1f}% of test layouts have a one-buoy twin in training')

# example near-twin pair inside the best layout's family, move close to the median
pf = pairs[(fam[pairs.a.to_numpy()] == fam[best])].copy()
pf['gap'] = (pf.moved - 75.0).abs()   # the pair shown in the EC3 deck (75 m move); under the true median it is a small move
ex = pf.sort_values('gap').iloc[0]
a, b = int(ex.a), int(ex.b)
ka = {(round(x, 3), round(y, 3)) for x, y in zip(UX[a], UY[a])}
kb = {(round(x, 3), round(y, 3)) for x, y in zip(UX[b], UY[b])}
print(f'example pair: moved {ex.moved:.0f} m, power {tp[a]/1e3:.1f} vs {tp[b]/1e3:.1f} kW, shared buoys {len(ka & kb)}')

# top-5% layouts for the position map
top = np.argsort(-tp)[: max(1, nu // 20)]
out = {
    'rows': rows, 'unique': nu, 'dup_pct': round(100 * (1 - nu / rows), 1),
    'p_iso': P_ISO, 'theta_deg': 55.3, 'corr_power_u': round(float(corr), 3),
    'thirds_kW': [round(front / 1e3, 1), round(mid / 1e3, 1), round(back / 1e3, 1)],
    'shadow_kW': [round(g[~shadow].mean() / 1e3, 1), round(g[shadow].mean() / 1e3, 1)],
    'pairs': int(len(pairs)), 'moved_median_m': round(float(pairs.moved.median()), 0),
    'families': int(ncomp), 'largest_family': int(sizes.max()), 'best_family_size': int(sizes[fam[best]]),
    'twin_share_pct': round(100 * float(has_twin.mean()), 1), 'leak_share_pct': round(100 * float(leak.mean()), 1),
    'best': {'x': UX[best].round(2).tolist(), 'y': UY[best].round(2).tolist(), 'p': g[best].round(0).tolist(), 'total': round(float(tp[best]), 0)},
    'pair': {'a': {'x': UX[a].round(2).tolist(), 'y': UY[a].round(2).tolist(), 'p': g[a].round(0).tolist(), 'total': round(float(tp[a]), 0)},
             'b': {'x': UX[b].round(2).tolist(), 'y': UY[b].round(2).tolist(), 'p': g[b].round(0).tolist(), 'total': round(float(tp[b]), 0)},
             'moved_m': round(float(ex.moved), 0)},
    'top_u': (U[top] - U[top].mean(axis=1, keepdims=True)).round(1).tolist(),
    'top_v': (V[top] - V[top].mean(axis=1, keepdims=True)).round(1).tolist(),
    'top_p': g[top].round(0).tolist(),
}
json.dump(out, open(f'{OUT}/uci_geometry.json', 'w'))
np.savez_compressed(f'{OUT}/uci_unique.npz', X=UX, Y=UY, P=g, T=tp, cnt=cnt, fam=fam, pa=pairs.a.to_numpy(), pb=pairs.b.to_numpy(), moved=pairs.moved.to_numpy())
print('saved')
