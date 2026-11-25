"""
Learn the sheltering threshold: classify where ERA5 is UNTRUSTWORTHY.

The regression -- predict the size of ERA5's bias from site features -- failed
(per-station R2 between -0.17 and -0.58). This asks the coarser question:

    is |relative bias| at this site above a threshold, yes or no?

Features are GEOMETRY ONLY. No ERA5 variables, because at an unmeasured cell
the map is all you have.

Validated leave-one-station-out and leave-one-region-out. The second is the
one that matters: it says whether the flag exports to a coastline the model
has never seen.

    python shelter_classifier.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.20]
GEOM_HINTS = ["slope", "dist_coast_km", "fetch_mean", "fetch_min", "fetch_max",
              "fetch_median", "fetch_p25", "fetch_aniso",
              "blocked_10km", "blocked_25km", "blocked_50km",
              "open_5km", "open_10km", "open_25km", "open_50km", "open_100km"]


def load():
    df = pd.read_csv("pairs.csv")
    geom = [c for c in GEOM_HINTS if c in df.columns]
    geom += [c for c in df.columns if c.startswith("rose_")]

    agg = {c: "first" for c in geom + ["region", "lat", "lon", "depth_m"]}
    per = (df.groupby("station")
             .agg({**agg, "resid_hs": "mean", "WVHT": "mean",
                   "era5_swh": "mean"})
             .reset_index())
    per["rel_bias"] = per.resid_hs / per.era5_swh
    per["abs_rel"] = per.rel_bias.abs()
    return per, geom


def models():
    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000,
                                                     class_weight="balanced")),
        "random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=0),
    }


def loso_auc(X, y, mdl):
    """Out-of-fold AUC, stratified so each fold sees both classes."""
    if y.sum() < 3 or (~y.astype(bool)).sum() < 3:
        return np.nan
    oof = np.zeros(len(y))
    cv = StratifiedKFold(min(5, int(y.sum())), shuffle=True, random_state=0)
    for tr, te in cv.split(X, y):
        m = mdl()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def loro_auc(X, y, groups, mdl):
    """Train on all regions but one, score the held-out region. Pooled AUC."""
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if y[tr].sum() < 2:
            continue
        m = mdl()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], oof[ok])


def main():
    per, geom = load()
    X = per[geom].values
    groups = per.region.values
    print(f"{len(per)} stations, {len(geom)} geometry features\n")

    print("=" * 66)
    print("AUC for 'is ERA5 untrustworthy here?'   (0.5 = coin flip)")
    print("=" * 66)
    print(f"{'threshold':>10s} {'n_flagged':>10s} "
          f"{'logistic LOSO':>14s} {'RF LOSO':>9s} "
          f"{'logistic LORO':>14s} {'RF LORO':>9s}")

    best = (None, -1)
    for t in THRESHOLDS:
        y = (per.abs_rel > t).astype(int).values
        if y.sum() < 4:
            print(f"{t:10.2f} {y.sum():10d}   (too few flagged)")
            continue
        row = []
        for name, _ in models().items():
            mk = (lambda n=name: models()[n])
            row.append(loso_auc(X, y, mk))
        for name, _ in models().items():
            mk = (lambda n=name: models()[n])
            row.append(loro_auc(X, y, groups, mk))
        print(f"{t:10.2f} {y.sum():10d} {row[0]:14.3f} {row[1]:9.3f} "
              f"{row[2]:14.3f} {row[3]:9.3f}")
        score = np.nanmean([row[2], row[3]])
        if np.isfinite(score) and score > best[1]:
            best = (t, score)

    if best[0] is None:
        print("\nnothing usable")
        return

    t = best[0]
    y = (per.abs_rel > t).astype(int).values
    print(f"\nbest threshold: |relative bias| > {t:.2f}  "
          f"({y.sum()} of {len(y)} stations flagged)")

    # ------------------------------------------------- what drives the flag
    rf = models()["random forest"]
    rf.fit(X, y)
    imp = permutation_importance(rf, X, y, n_repeats=20, random_state=0,
                                 scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1][:8]
    print("\n--- what the classifier uses ---")
    for i in order:
        print(f"  {geom[i]:16s} {imp.importances_mean[i]:+.4f}")

    # ------------------------------------------------------ flagged stations
    per["p_flag"] = rf.predict_proba(X)[:, 1]
    print("\n--- stations the classifier flags most strongly ---")
    top = per.nlargest(12, "p_flag")
    print(top[["station", "region", "depth_m", "fetch_aniso", "blocked_25km",
               "rel_bias", "p_flag"]].round(3).to_string(index=False))

    print("\n--- actual worst ERA5 errors, and whether they were flagged ---")
    worst = per.nlargest(10, "abs_rel")
    print(worst[["station", "region", "rel_bias", "fetch_aniso",
                 "blocked_25km", "p_flag"]].round(3).to_string(index=False))

    per.to_csv("shelter_flags.csv", index=False)
    print(f"\nwritten to shelter_flags.csv")
    print("\nVERDICT:", "usable filter" if best[1] > 0.70 else
          "weak -- geometry does not reliably flag untrustworthy cells")


if __name__ == "__main__":
    main()
