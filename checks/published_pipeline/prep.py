# Prep: unique layouts, duplicate checks, independent 48-of-49 families.
# usage: python3 -I prep.py <csv> <out_npz> [team_npz]
import sys
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CSV, OUT = sys.argv[1], sys.argv[2]
TEAM = sys.argv[3] if len(sys.argv) > 3 else None
N = 49
df = pd.read_csv(CSV)
X = df[[f'X{i}' for i in range(1, N + 1)]].to_numpy(float)
Y = df[[f'Y{i}' for i in range(1, N + 1)]].to_numpy(float)
PW = df[[f'Power{i}' for i in range(1, N + 1)]].to_numpy(float)
TP = df['Total_Power'].to_numpy(float)
QW = df['qW'].to_numpy(float)
rows = len(df)
print('rows', rows, 'cols', df.shape[1], 'NaN', int(df.isna().sum().sum()))
print('coord range x', X.min(), X.max(), 'y', Y.min(), Y.max())
print('Total_Power min/mean/max MW', TP.min() / 1e6, TP.mean() / 1e6, TP.max() / 1e6)
d = TP - PW.sum(1)
print('Total_Power - sum(Power_i): max abs', np.abs(d).max(), 'W; rows with |diff|>1W:', int((np.abs(d) > 1).sum()))
print('qW vs TP: corr', np.corrcoef(QW, TP)[0, 1], ' TP/(N*qW) min/max', (TP / (N * QW)).min(), (TP / (N * QW)).max())

# exact duplicates, buoy order kept (same rule as the team)
coords = np.round(np.concatenate([X, Y], axis=1), 6)
uniq, first, inv = np.unique(coords, axis=0, return_index=True, return_inverse=True)
inv = inv.ravel(); nu = len(uniq)
cnt = np.bincount(inv)
print('unique (ordered)', nu, 'dup%', round(100 * (1 - nu / rows), 1))
print('copies per layout: max', cnt.max(), 'median', np.median(cnt), 'layouts with >1 copy', int((cnt > 1).sum()),
      'top5 counts', np.sort(cnt)[-5:])
s = pd.Series(TP).groupby(inv)
sd = s.std(ddof=0).fillna(0).to_numpy(); rng_ = (s.max() - s.min()).to_numpy()
print('Total_Power spread inside exact-duplicate groups: max range W', rng_.max(), 'groups with range>1W', int((rng_ > 1).sum()),
      'max sd', sd.max())
tp_u = s.mean().sort_index().to_numpy()
UX, UY = uniq[:, :N], uniq[:, N:]

# order-free duplicates (same set of positions, any buoy order)
key = np.round(UX * 1000).astype(np.int64) * 10_000_000 + np.round(UY * 1000).astype(np.int64)
ks = np.sort(key, axis=1)
uq2, inv2 = np.unique(ks, axis=0, return_inverse=True)
print('unique as unordered position sets', len(uq2))
# buoys sharing a position inside a layout
print('layouts with a repeated position inside', int((np.diff(ks, axis=1) == 0).any(1).sum()))

# independent 48-of-49: exact tuples of the sorted remaining 48 keys (0.1 m rounding like the team, and 1 mm)
def families(res):
    k = np.round(UX * res).astype(np.int64) * 100_000_000 + np.round(UY * res).astype(np.int64)
    k = np.sort(k, axis=1)
    seen = {}
    ea, eb = [], []
    for li in range(nu):
        r = k[li]
        for j in range(N):
            t = np.delete(r, j).tobytes()
            lst = seen.get(t)
            if lst is None:
                seen[t] = [li]
            else:
                for o in lst:
                    if o != li:
                        ea.append(o); eb.append(li)
                if lst[-1] != li:
                    lst.append(li)
    a = np.minimum(ea, eb); b = np.maximum(ea, eb)
    pr = np.unique(np.stack([a, b], 1), axis=0) if len(ea) else np.zeros((0, 2), int)
    # drop pairs that are the same set (49 of 49 shared, different order) -> would be distance-0 pairs
    same = (k[pr[:, 0]] == k[pr[:, 1]]).all(1)
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(nu, nu))
    nc, fam = connected_components(A, directed=False)
    return pr, same, nc, fam

for res in (10, 1000):
    pr, same, nc, fam = families(res)
    sz = np.bincount(fam)
    best = int(np.argmax(tp_u))
    print(f'[res 1/{res} m] pairs {len(pr)} (of which identical sets {int(same.sum())}), families {nc}, singletons {(sz==1).sum()}, '
          f'largest {sz.max()}, best-layout family {sz[fam[best]]}, sizes top5 {np.sort(sz)[-5:]}')
    if res == 10:
        fam10, pr10 = fam, pr

if TEAM:
    t = np.load(TEAM)
    print('team npz: X equal', np.array_equal(t['X'], UX), 'T equal', np.allclose(t['T'], tp_u), 'cnt equal', np.array_equal(t['cnt'], cnt))
    # same partition?
    tf = t['fam']
    m1 = pd.Series(tf).groupby(fam10).nunique().max(); m2 = pd.Series(fam10).groupby(tf).nunique().max()
    print('team fam vs mine: same partition', bool(m1 == 1 and m2 == 1), 'team pairs', len(t['pa']), 'mine', len(pr10))

# per-family power stats
sz = np.bincount(fam10)
best = int(np.argmax(tp_u)); ef = fam10[best]
em = fam10 == ef
print('elite family: n', int(em.sum()), 'rows', int(cnt[em].sum()), 'TP mean kW', tp_u[em].mean() / 1e3, 'min', tp_u[em].min() / 1e3,
      'max', tp_u[em].max() / 1e3, '| all layouts mean', tp_u.mean() / 1e3, 'sd', tp_u.std() / 1e3,
      '| rows mean', TP.mean() / 1e3, 'rows sd', TP.std() / 1e3)
print('elite share of top-5% layouts', float(em[np.argsort(-tp_u)[: nu // 20]].mean()))
np.savez_compressed(OUT, UX=UX, UY=UY, tp=tp_u, cnt=cnt, inv=inv, TP=TP, fam=fam10, pa=pr10[:, 0], pb=pr10[:, 1], PWu=pd.DataFrame(PW).groupby(inv).mean().sort_index().to_numpy())
print('saved', OUT)
