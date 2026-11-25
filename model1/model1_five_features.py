#!/usr/bin/env python3
"""H: station_summary label, distance-to-coast AUC, region counts, LORO RF/LR on available features.
usage: 08_model1.py <station_summary.csv>"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.simplefilter("ignore")
s = pd.read_csv(sys.argv[1])
print("rows", len(s), "unique stations", s.station.nunique(), "regions", s.region.value_counts().to_dict())
print("check rel_bias == bias/era5_mean:", np.allclose(s.rel_bias, s.bias / s.era5_mean),
      " bias == buoy-era5:", np.allclose(s.bias, s.buoy_mean - s.era5_mean, atol=1e-6))
defs = {"|rel_bias|>0.08": s.rel_bias.abs() > 0.08, "|bias/buoy_mean|>0.08": (s.bias / s.buoy_mean).abs() > 0.08,
        "|bias_dm|>0.08": s.bias_dm.abs() > 0.08, "|rel_bias|>=0.08": s.rel_bias.abs() >= 0.08}
for k, y in defs.items():
    print("%-24s positives %d / %d ; by region %s" % (k, y.sum(), len(y), s.groupby("region")[y.name if False else []].size().to_dict() if False else
          {r: "%d/%d" % (y[s.region == r].sum(), (s.region == r).sum()) for r in sorted(s.region.unique())}))
y = (s.rel_bias.abs() > 0.08).astype(int).values
print("\nunfitted AUCs for y=|rel_bias|>0.08 (score sign chosen so that smaller value = more risk where noted):")
for f in ["coast", "depth", "log_depth", "slope", "fetch"]:
    a = roc_auc_score(y, -s[f].values)
    print("  -%s: %.3f   (+%s: %.3f)" % (f, a, f, 1 - a))
for k, yy in defs.items():
    print("  -coast AUC under %s: %.3f" % (k, roc_auc_score(yy.astype(int), -s.coast.values)))
feats = ["depth", "coast", "slope", "fetch", "log_depth"]
X = s[feats].values
reg = s.region.values
def loro(model):
    p = np.zeros(len(y))
    for r in np.unique(reg):
        tr, te = reg != r, reg == r
        model.fit(X[tr], y[tr]); p[te] = model.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)
print("\nLORO (pooled out-of-region probabilities), features", feats)
print("  LR: %.3f" % loro(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))))
rf = [loro(RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=sd, n_jobs=1)) for sd in range(5)]
print("  RF (5 seeds): %s mean %.3f" % (np.round(rf, 3), np.mean(rf)))
# coast only LORO LR (monotone so same as unfitted pooled? check)
X = s[["coast"]].values
print("  LR coast only: %.3f" % loro(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))))
