"""
WEC farm dataset audit: duplicate layouts, target leakage, and transfer failure.
Run from a directory containing WEC/WEC_{Perth,Sydney}_{49,100}.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

FILES = [("WEC_Perth_49", 49), ("WEC_Sydney_49", 49),
         ("WEC_Perth_100", 100), ("WEC_Sydney_100", 100)]


def coord_cols(df):
    return [c for c in df.columns if c[0] in "XY" and c[1:].isdigit()]


def power_cols(df):
    return [c for c in df.columns if c.startswith("Power")]


# ---------------------------------------------------------------- 1. duplicates
def audit_duplicates():
    print("=" * 68)
    print("1. DUPLICATE LAYOUTS")
    print("=" * 68)
    for name, n in FILES:
        df = pd.read_csv(f"WEC/{name}.csv")
        co = coord_cols(df)
        u = len(df[co].drop_duplicates())
        print(f"{name:18s} rows={len(df):6d}  unique={u:6d}  "
              f"duplicated={100*(1-u/len(df)):5.1f}%")


# ------------------------------------------------------------------ 2. sum leak
def audit_target_leak(name="WEC_Perth_49"):
    print("\n" + "=" * 68)
    print("2. TARGET LEAKAGE: Total_Power is the sum of Power_i")
    print("=" * 68)
    df = pd.read_csv(f"WEC/{name}.csv")
    co, P = coord_cols(df), power_cols(df)
    g = df.groupby(co, sort=False).ngroup().values
    r = cross_val_score(LinearRegression(), df[P].values, df["Total_Power"].values,
                        cv=GroupKFold(5), groups=g, scoring="r2")
    print(f"LinearRegression(Power_i) -> Total_Power   R2 = {r.mean():.8f}")
    print("Any model using Power_i as a feature is doing arithmetic, not learning.")


# -------------------------------------------------- 3. random vs grouped split
def audit_split(name="WEC_Perth_49"):
    print("\n" + "=" * 68)
    print("3. RANDOM vs GROUPED-BY-LAYOUT CV  (coordinates only)")
    print("=" * 68)
    df = pd.read_csv(f"WEC/{name}.csv")
    co = coord_cols(df)
    X, y = df[co].values, df["Total_Power"].values
    g = df.groupby(co, sort=False).ngroup().values
    mk = lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0)
    a = cross_val_score(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    b = cross_val_score(mk(), X, y, cv=GroupKFold(5), groups=g, scoring="r2")
    print(f"random 5-fold        R2 = {a.mean():.4f}")
    print(f"grouped-by-layout    R2 = {b.mean():.4f}")
    print(f"inflation            {a.mean()-b.mean():+.4f}")


# ---------------------------------------------- 4. permutation-invariant feats
def invariant_features(df, n):
    """Summarise a layout by statistics that don't depend on WEC ordering."""
    X = df[[f"X{i}" for i in range(1, n + 1)]].values
    Y = df[[f"Y{i}" for i in range(1, n + 1)]].values
    rows = []
    for x, y in zip(X, Y):
        p = np.c_[x, y]
        d = pdist(p)
        full = np.linalg.norm(p[:, None] - p[None], axis=2)
        np.fill_diagonal(full, np.inf)
        nn = full.min(axis=1)
        rows.append([d.mean(), d.min(), np.percentile(d, 10), np.percentile(d, 90),
                     d.std(), nn.mean(), nn.min(), nn.std(),
                     x.std(), y.std(), x.max() - x.min(), y.max() - y.min()])
    return np.array(rows)


def audit_transfer(sample=1500):
    print("\n" + "=" * 68)
    print("4. TRANSFER: invariant features -> power per WEC")
    print("=" * 68)
    sets = {}
    for name, n in FILES:
        df = pd.read_csv(f"WEC/{name}.csv")
        df = df.drop_duplicates(subset=coord_cols(df)).reset_index(drop=True)
        sub = df.sample(min(sample, len(df)), random_state=0)
        sets[name] = (invariant_features(sub, n), sub["Total_Power"].values / n)

    mk = lambda: HistGradientBoostingRegressor(max_iter=400, random_state=0)
    pairs = [("WEC_Perth_49", "WEC_Perth_49"),
             ("WEC_Perth_49", "WEC_Sydney_49"),
             ("WEC_Perth_49", "WEC_Perth_100"),
             ("WEC_Sydney_49", "WEC_Sydney_100")]
    for tr, te in pairs:
        Xa, ya = sets[tr]
        Xb, yb = sets[te]
        if tr == te:
            k = len(Xa) // 2
            pred = mk().fit(Xa[:k], ya[:k]).predict(Xa[k:])
            truth = ya[k:]
        else:
            pred = mk().fit(Xa, ya).predict(Xb)
            truth = yb
        # separate offset from shape
        bias = pred.mean() - truth.mean()
        r2 = r2_score(truth, pred)
        r2_deb = r2_score(truth, pred - bias)
        print(f"train {tr:16s} test {te:16s} "
              f"R2 = {r2:9.3f}   debiased R2 = {r2_deb:7.3f}   bias = {bias:9.1f}")


if __name__ == "__main__":
    audit_duplicates()
    audit_target_leak()
    audit_split()
    audit_transfer()
