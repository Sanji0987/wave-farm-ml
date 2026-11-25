#!/usr/bin/env python3
"""Load the four raw CSVs with exact decimal handling; save npz caches.
Usage: python3 -I s01_load.py <dir with CSVs> <out dir>
"""
import sys, os, csv, re
from decimal import Decimal
import numpy as np

src, out = sys.argv[1], sys.argv[2]
FILES = {"Perth_49": "WEC_Perth_49.csv", "Sydney_49": "WEC_Sydney_49.csv",
         "Perth_100": "WEC_Perth_100.csv", "Sydney_100": "WEC_Sydney_100.csv"}

for key, fn in FILES.items():
    path = os.path.join(src, fn)
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = [r for r in rd]
    n = len(rows)
    ncol = len(hdr)
    bad_len = sum(1 for r in rows if len(r) != ncol)
    N = sum(1 for h in hdr if re.fullmatch(r"X\d+", h))
    xi = [hdr.index(f"X{i}") for i in range(1, N + 1)]
    yi = [hdr.index(f"Y{i}") for i in range(1, N + 1)]
    pi = [hdr.index(f"Power{i}") for i in range(1, N + 1)]
    qi = hdr.index("qW"); ti = hdr.index("Total_Power")
    # decimal-precision audit of coordinate strings
    maxdec = 0; nonnum = 0; sci = 0
    decs = {}
    for r in rows:
        for j in xi + yi:
            s = r[j].strip()
            if "e" in s.lower():
                sci += 1
            try:
                d = Decimal(s)
            except Exception:
                nonnum += 1
                continue
            e = -d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0
            decs[e] = decs.get(e, 0) + 1
            if e > maxdec:
                maxdec = e
    # exact integer representation at 10^-maxdec m
    scale = 10 ** max(maxdec, 2)
    X = np.empty((n, N), dtype=np.int64); Y = np.empty((n, N), dtype=np.int64)
    for a, r in enumerate(rows):
        for b, j in enumerate(xi):
            X[a, b] = int(Decimal(r[j].strip()) * scale)
        for b, j in enumerate(yi):
            Y[a, b] = int(Decimal(r[j].strip()) * scale)
    P = np.array([[float(r[j]) for j in pi] for r in rows], dtype=np.float64)
    q = np.array([float(r[qi]) for r in rows]); T = np.array([float(r[ti]) for r in rows])
    # decimals of qW and powers
    qdec = {}
    for r in rows:
        s = r[qi].strip(); e = len(s.split(".")[1]) if "." in s else 0
        qdec[e] = qdec.get(e, 0) + 1
    pdec = {}
    for r in rows[:2000]:
        for j in pi + [ti]:
            s = r[j].strip(); e = len(s.split(".")[1]) if "." in s else 0
            pdec[e] = pdec.get(e, 0) + 1
    print(f"{key}: rows={n} ncol={ncol} N={N} bad_len={bad_len} nonnum={nonnum} sci={sci} "
          f"coord_decimals={dict(sorted(decs.items()))} scale={scale} qW_decimals={qdec} power_decimals(first2000)={pdec}")
    print(f"   X min/max = {X.min()/scale} / {X.max()/scale};  Y min/max = {Y.min()/scale} / {Y.max()/scale}")
    print(f"   NaN P={np.isnan(P).sum()} q={np.isnan(q).sum()} T={np.isnan(T).sum()};  qW range {q.min()}..{q.max()}; T range {T.min()}..{T.max()}")
    np.savez_compressed(os.path.join(out, f"{key}.npz"), X=X, Y=Y, P=P, q=q, T=T, scale=scale)
