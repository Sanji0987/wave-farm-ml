# Linear vs quadratic vs boosted trees on the 40 per-buoy features, whole twin families held out (GroupKFold, 5 folds).
# usage: python3 -I model_comparison.py <out_dir from train_perbuoy.py>   (needs uci_unique.npz and feats.npy there)
# Results on 17 Nov 2026 (full data): linear R2 0.846, quadratic 0.932 (MAE 23.5 kW), trees 0.961 (MAE 16.2 kW);
# error on the true top 5%: 79.1 vs 78.6 kW. Raw coordinates, farm level: linear 0.671, trees 0.711 (train 0.987).
import sys, numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
O = sys.argv[1]
d = np.load(f'{O}/uci_unique.npz'); X, Y, P, T, fam = d['X'], d['Y'], d['P'], d['T'], d['fam']
F = np.load(f'{O}/feats.npy').astype(np.float32); nu, N = X.shape; PISO = 96374.0
r = (P / PISO).ravel(); lay = np.repeat(np.arange(nu), N)
folds = list(GroupKFold(5).split(F, r, fam[lay]))
top = T >= np.quantile(T, 0.95)
models = {
    'linear': lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
    'quadratic': lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(10.0)),
    'trees': lambda: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
                                                   l2_regularization=1.0, early_stopping=False, random_state=0)}
for name, mk in models.items():
    oof = np.zeros_like(r)
    for tr, te in folds: oof[te] = mk().fit(F[tr], r[tr]).predict(F[te])
    pf = PISO * oof.reshape(nu, N).sum(1); e = pf - T
    print(f'{name:9s} farm R2 {1 - np.sum(e**2) / np.sum((T - T.mean())**2):.3f}  MAE {np.mean(np.abs(e))/1e3:.1f} kW  '
          f'rho {spearmanr(T, pf)[0]:.3f}  MAE on true top 5% {np.mean(np.abs(e[top]))/1e3:.1f} kW')
Xr = np.c_[X, Y]
for name, mk in [('raw linear', lambda: make_pipeline(StandardScaler(), Ridge(1.0))),
                 ('raw trees', lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0))]:
    oof = np.zeros(nu)
    for tr, te in GroupKFold(5).split(Xr, T, fam): oof[te] = mk().fit(Xr[tr], T[tr]).predict(Xr[te])
    print(f'{name:10s} farm R2 {1 - np.sum((T - oof)**2) / np.sum((T - T.mean())**2):.3f}')
