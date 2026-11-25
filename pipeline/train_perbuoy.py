# Per-buoy surrogate for the demo: features from each buoy's neighbourhood -> buoy power / isolated power.
# usage: python3 -I train_perbuoy.py <out_dir> [cv]
import sys, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

OUT = sys.argv[1]; DO_CV = len(sys.argv) > 2 and sys.argv[2] == 'cv'
d = np.load(f'{OUT}/uci_unique.npz')
X, Y, P, T, fam = d['X'], d['Y'], d['P'], d['T'], d['fam']
nu, N = X.shape
P_ISO = 96374.0
TH = np.deg2rad(55.3); C, S = np.cos(TH), np.sin(TH)
K = 12; RADII = [150, 300, 600, 1200]; CAP = 2000.0

def feats(x, y):
    """x, y: (49,) arrays -> (49, 40) features. Mirrors the JS implementation exactly."""
    u = x * C + y * S; v = -x * S + y * C
    du = u[None, :] - u[:, None]; dv = v[None, :] - v[:, None]   # du[i, j] = u_j - u_i
    dd = np.sqrt(du * du + dv * dv)
    n = len(x); F = np.zeros((n, 2 * K + 8 + 3 + 1 + 2 + 2))
    um, vm = u.mean(), v.mean()
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        di, ui, vi = dd[i, idx], du[i, idx], dv[i, idx]
        o = np.argsort(di, kind='stable')[:K]
        F[i, 0:2 * K:2] = ui[o]; F[i, 1:2 * K:2] = vi[o]
        a = np.abs(np.degrees(np.arctan2(vi, ui)))
        up, dn, sd = a <= 30, a >= 150, (a >= 60) & (a <= 120)
        col = 2 * K
        for R in RADII: F[i, col] = np.sum(up & (di <= R)); col += 1
        for R in RADII: F[i, col] = np.sum(dn & (di <= R)); col += 1
        for msk in (up, dn, sd):
            F[i, col] = min(CAP, di[msk].min()) if msk.any() else CAP; col += 1
        F[i, col] = np.sum(ui > 0) / (n - 1); col += 1
        F[i, col] = u[i] - um; F[i, col + 1] = v[i] - vm; col += 2
        F[i, col] = np.sum(di <= 150); F[i, col + 1] = np.sum(di <= 300)
    return F

NAMES = [f'{a}{k+1}' for k in range(K) for a in ('du', 'dv')] + [f'up{R}' for R in RADII] + [f'dn{R}' for R in RADII] + \
        ['nn_up', 'nn_dn', 'nn_side', 'frac_front', 'u_c', 'v_c', 'dens150', 'dens300']

t0 = time.time()
F = np.concatenate([feats(X[k], Y[k]) for k in range(nu)])
r = (P / P_ISO).ravel()
lay = np.repeat(np.arange(nu), N)
print(f'features {F.shape} in {time.time()-t0:.0f}s')
np.save(f'{OUT}/feats.npy', F.astype(np.float64))

def model(it=300):
    return HistGradientBoostingRegressor(max_iter=it, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)

def farm_scores(pred_r, mask_lay):
    pt = P_ISO * pred_r.reshape(nu, N).sum(1); tt = T
    m = mask_lay
    res = tt[m] - pt[m]
    r2 = 1 - np.sum(res ** 2) / np.sum((tt[m] - tt[m].mean()) ** 2)
    return r2, float(spearmanr(tt[m], pt[m])[0]), float(np.mean(np.abs(res)) / 1e3)

if DO_CV:
    res = {}
    best_fam = fam[np.argmax(T)]
    for split, groups in [('exact', lay), ('family', fam[lay])]:
        oof = np.zeros_like(r)
        for tr, te in GroupKFold(5).split(F, r, groups):
            m = model(250).fit(F[tr], r[tr]); oof[te] = m.predict(F[te])
            print(split, 'fold done', f'{time.time()-t0:.0f}s', flush=True)
        pb = 1 - np.sum((r - oof) ** 2) / np.sum((r - r.mean()) ** 2)
        r2, rho, mae = farm_scores(oof, np.ones(nu, bool))
        el = fam == best_fam
        bias_el = float(np.mean((P_ISO * oof.reshape(nu, N).sum(1) - T)[el]) / 1e3 / N)
        res[split] = {'perbuoy_r2': round(float(pb), 3), 'farm_r2': round(float(r2), 3), 'farm_rho': round(rho, 3),
                      'farm_mae_kW': round(mae, 1), 'elite_bias_kW_per_buoy': round(bias_el, 2)}
        print(split, res[split], flush=True)
    json.dump(res, open(f'{OUT}/cv_scores.json', 'w'), indent=1)

# final model: hold out ~15% of families (never the best layout's family) as honest examples for the demo
rng = np.random.default_rng(1)
fams = np.unique(fam); best_fam = fam[np.argmax(T)]
hold = set(rng.choice(fams[fams != best_fam], size=int(0.15 * len(fams)), replace=False).tolist())
hold_lay = np.array([f in hold for f in fam])
trm = ~hold_lay[lay]
m = model(300).fit(F[trm], r[trm])
pred = m.predict(F)
r2h, rhoh, maeh = farm_scores(pred, hold_lay)
print(f'final model, held-out families: farm R2 {r2h:.3f}, rho {rhoh:.3f}, MAE {maeh:.1f} kW, n={hold_lay.sum()}')
import pickle
pickle.dump({'model': m, 'hold_lay': hold_lay, 'names': NAMES}, open(f'{OUT}/final_model.pkl', 'wb'))

# export trees
trees = []
for it in m._predictors:
    nd = it[0].nodes
    trees.append({'f': nd['feature_idx'].astype(int).tolist(), 't': [float(v) for v in nd['num_threshold']],
                  'l': nd['left'].astype(int).tolist(), 'r': nd['right'].astype(int).tolist(),
                  'v': [round(float(v), 7) for v in nd['value']], 'leaf': nd['is_leaf'].astype(int).tolist()})
base = float(np.ravel(m._baseline_prediction)[0])
# check manual evaluation == sklearn
def manual(Fm):
    out = np.full(len(Fm), base)
    for tr in trees:
        for q in range(len(Fm)):
            n_ = 0
            while not tr['leaf'][n_]:
                n_ = tr['l'][n_] if Fm[q, tr['f'][n_]] <= tr['t'][n_] else tr['r'][n_]
            out[q] += tr['v'][n_]
    return out
chk = manual(F[:200]); print('manual vs sklearn max abs diff', float(np.max(np.abs(chk - m.predict(F[:200])))))
json.dump({'base': base, 'trees': trees, 'names': NAMES, 'p_iso': P_ISO, 'theta_deg': 55.3, 'K': K, 'radii': RADII, 'cap': CAP,
           'holdout_scores': {'farm_r2': round(float(r2h), 3), 'farm_rho': round(rhoh, 3), 'farm_mae_kW': round(maeh, 1), 'n_layouts': int(hold_lay.sum())}},
          open(f'{OUT}/model_trees.json', 'w'), separators=(',', ':'))
np.save(f'{OUT}/pred_final.npy', pred)
print('exported', len(trees), 'trees')
