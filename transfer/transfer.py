# Transfer tests for the per-buoy model (results.md §5): train on one file, predict another it has never seen.
# usage: python3 -I transfer.py <dir with WEC_*.csv> <out.json>
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import load_unique, feats_all, model, farm_metrics, fit_theta, P_ISO

D, OUT = sys.argv[1], sys.argv[2]
t0 = time.time()
data = {}
for site in ('Perth', 'Sydney'):
    for N in (49, 100):
        X, Y, P, T = load_unique(f'{D}/WEC_{site}_{N}.csv')
        q = pd.read_csv(f'{D}/WEC_{site}_{N}.csv', usecols=['qW', 'Total_Power'])
        piso = float(np.median(q.Total_Power / (N * q.qW)))
        data[(site, N)] = (X, Y, P, T)
        print(f'{site} {N}: {len(X)} unique layouts, lone-buoy power implied by qW {piso:,.0f} W', flush=True)

theta = {}
for site in ('Perth', 'Sydney'):
    X, Y, P, T = data[(site, 49)]
    th, c = fit_theta(X, Y, P)
    theta[site] = th
    print(f'{site}: best-fit wave direction {th:.1f} deg (mean corr {c:.3f})', flush=True)
theta_used = {'Perth': 55.3, 'Sydney': theta['Sydney']}   # Perth keeps the team's 55.3

cache = {}
def F_of(site, N, th):
    k = (site, N, th)
    if k not in cache:
        X, Y, _, _ = data[(site, N)]
        cache[k] = feats_all(X, Y, th)
    return cache[k]

def run(src, dst, th_src, th_dst, label):
    (s_site, s_N), (d_site, d_N) = src, dst
    Xs, Ys, Ps, Ts = data[src]
    m = model(300).fit(F_of(s_site, s_N, th_src), (Ps / P_ISO[s_site]).ravel())
    Xd, Yd, Pd, Td = data[dst]
    pred = P_ISO[d_site] * m.predict(F_of(d_site, d_N, th_dst)).reshape(len(Xd), d_N).sum(1)
    r = farm_metrics(Td, pred); r['method'] = label
    print(f'{s_site} {s_N} -> {d_site} {d_N} [{label}]: {r}  ({time.time()-t0:.0f}s)', flush=True)
    return r

res = {'theta_fit': theta, 'theta_used': theta_used, 'runs': []}
def add(src, dst, th_s, th_d, label):
    r = run(src, dst, th_s, th_d, label); r['train'] = f'{src[0]} {src[1]}'; r['test'] = f'{dst[0]} {dst[1]}'
    res['runs'].append(r)

add(('Perth', 49), ('Perth', 100), 55.3, 55.3, 'same site')
add(('Sydney', 49), ('Sydney', 100), theta_used['Sydney'], theta_used['Sydney'], 'same site')
add(('Perth', 49), ('Sydney', 49), 55.3, theta_used['Sydney'], 'each site in its own wave frame')
add(('Perth', 49), ('Sydney', 49), 55.3, 55.3, 'Perth wave frame for both')
add(('Sydney', 49), ('Perth', 49), theta_used['Sydney'], 55.3, 'each site in its own wave frame')
add(('Sydney', 49), ('Perth', 49), theta_used['Sydney'], theta_used['Sydney'], 'Sydney wave frame for both')
json.dump(res, open(OUT, 'w'), indent=1)
print('saved', OUT)
