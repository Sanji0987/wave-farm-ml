#!/usr/bin/env python3
"""F follow-up: which set/distance gives 114-167? usage: 06b_independent_grid.py <cells_with_wave.pkl>"""
import sys
import numpy as np, pandas as pd
df = pd.read_pickle(sys.argv[1]); R = 6371.0088
def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
def greedy_km(sub, order, dmin):
    X = xyz(sub.lat.values, sub.lon.values)[order]; thr = 2*np.sin(dmin/(2*R)); acc = []
    for i in range(len(X)):
        if not acc or np.min(np.linalg.norm(X[acc]-X[i], axis=1)) >= thr: acc.append(i)
    return len(acc)
def greedy_deg(sub, order, dmin):
    X = np.c_[sub.lat.values, sub.lon.values][order]; acc = []
    for i in range(len(X)):
        if not acc or np.min(np.linalg.norm(X[acc]-X[i], axis=1)) >= dmin: acc.append(i)
    return len(acc)
sets = {"band+wave": df.band & df.has, "unflagged": df.band & df.has & ~df.flag_b}
for nm, m in sets.items():
    sub = df[m].reset_index(drop=True); o = np.argsort(-sub.P.values, kind="stable")
    r = np.random.default_rng(0).permutation(len(sub))
    out = []
    for dk in (40, 45, 50, 55, 60):
        out.append("km%d:%d/%d" % (dk, greedy_km(sub, o, dk), greedy_km(sub, r, dk)))
    for dg in (0.36, 0.4, 0.5):
        out.append("deg%.2f:%d/%d" % (dg, greedy_deg(sub, o, dg), greedy_deg(sub, r, dg)))
    print(nm, len(sub), " (descP/random seed0) ", " ".join(out))
