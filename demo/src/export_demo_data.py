# Compact model + layouts + map data for the offline demo, and a verification sample for the JS port.
# usage: python3 -I export_demo_data.py <out_dir> <demo_dir>
import sys, json, pickle, base64, io
import numpy as np, pandas as pd
from PIL import Image
OUT, DEMO = sys.argv[1], sys.argv[2]
M = pickle.load(open(f'{OUT}/final_model.pkl', 'rb')); m, hold = M['model'], M['hold_lay']
d = np.load(f'{OUT}/uci_unique.npz'); X, Y, P, T, fam = d['X'], d['Y'], d['P'], d['T'], d['fam']
pred = np.load(f'{OUT}/pred_final.npy')
F = np.load(f'{OUT}/feats.npy', mmap_mode='r')
G = json.load(open(f'{OUT}/uci_geometry.json'))
cv = json.load(open(f'{OUT}/cv_scores.json'))
# --- trees, full precision: leaf -> [v], split -> [f, t, l, r]
trees = []
for it in m._predictors:
    nd = it[0].nodes; tr = []
    for k in range(len(nd)):
        if nd['is_leaf'][k]: tr.append([float(nd['value'][k])])
        else: tr.append([int(nd['feature_idx'][k]), float(nd['num_threshold'][k]), int(nd['left'][k]), int(nd['right'][k])])
    trees.append(tr)
base = float(np.ravel(m._baseline_prediction)[0])
model = {'base': base, 'trees': trees, 'p_iso': 96374.0, 'C': 0.5692795234308443, 'S': 0.8221440410307373,
         'K': 12, 'radii': [150, 300, 600, 1200], 'cap': 2000.0}
json.dump(model, open(f'{DEMO}/model.json', 'w'), separators=(',', ':'))
# --- verification sample: 300 layouts (best + held-out + random)
rng = np.random.default_rng(7)
best = int(np.argmax(T))
ks = [best] + list(rng.choice(np.where(hold)[0], 150, replace=False)) + list(rng.choice(np.where(~hold)[0], 149, replace=False))
ver = {'x': X[ks].tolist(), 'y': Y[ks].tolist(), 'pred': pred.reshape(len(X), 49)[ks].tolist(),
       'feats0': np.asarray(F[best * 49:(best + 1) * 49]).tolist()}
json.dump(ver, open(f'{OUT}/verify_sample.json', 'w'))
# --- demo layouts
def lay(k, name, note):
    return {'name': name, 'note': note, 'x': X[k].tolist(), 'y': Y[k].tolist(),
            'sim_p': P[k].round(1).tolist(), 'sim_total': round(float(T[k]), 1), 'held_out': bool(hold[k])}
worst = int(np.argmin(T))
layouts = [lay(best, 'Best layout in the dataset', 'The highest simulated power of all 10,218 unique Perth layouts. The model trained on its family.'),
           lay(7759, 'Unseen layout A', 'From a layout family the demo model never saw in training.'),
           lay(5757, 'Unseen layout B', 'From a layout family the demo model never saw in training.'),
           lay(3056, 'Unseen layout C', 'From a layout family the demo model never saw in training.'),
           lay(2807, 'Unseen layout D', 'From a layout family the demo model never saw in training.'),
           lay(worst, 'Worst layout in the dataset', 'The lowest simulated power in the Perth 49-buoy file.')]
# exact coordinates (rounding to 3 dp changes features slightly; keep a flag and check both)
pairA, pairB = G['pair']['a'], G['pair']['b']
demo = {'layouts': layouts, 'cv': cv, 'n_unique': int(len(X)), 'rows': G['rows'], 'dup_pct': G['dup_pct'],
        'thirds_kW': G['thirds_kW'], 'leak_share_pct': G['leak_share_pct'], 'largest_family': G['largest_family'],
        'families': G['families'], 'pair': {'a': pairA, 'b': pairB, 'moved_m': G['pair']['moved_m']},
        'box': [0, 1000, 0, 1010]}
# --- map: land mask PNG + band cells
lm = np.load(f'{OUT}/landmask.npz'); land = lm['land']; la, lo = lm['lat'], lm['lon']
# crop to lon 70..95, lat -1..20, flip so north is up
sel_la = (la >= -1) & (la <= 20); sel_lo = (lo >= 70) & (lo <= 95)
sub = land[np.ix_(sel_la, sel_lo)][::-1]
img = np.zeros(sub.shape + (4,), np.uint8)
img[sub] = (0, 0, 0, 255)  # land opaque, water transparent: tinted in the page
buf = io.BytesIO(); Image.fromarray(img, 'RGBA').save(buf, format='PNG', optimize=True)
demo['map'] = {'png': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(),
               'lat': [float(la[sel_la].min()), float(la[sel_la].max())], 'lon': [float(lo[sel_lo].min()), float(lo[sel_lo].max())]}
b = pd.read_csv(f'{OUT}/band_cells.csv'); b = b[b.wpd.notna()]
demo['cells'] = [[round(r.lat, 4), round(r.lon, 4), round(r.wpd, 1), int(r.depth_m), round(r.dist_coast_km)] for r in b.itertuples()]
json.dump(demo, open(f'{DEMO}/demo_data.json', 'w'), separators=(',', ':'))
print('model.json', round(len(json.dumps(model, separators=(",", ":"))) / 1e3), 'KB; demo_data.json', round(len(json.dumps(demo, separators=(",", ":"))) / 1e3), 'KB;',
      'cells', len(demo['cells']), 'map px', sub.shape, 'worst', round(T[worst] / 1e3, 1), 'best', round(T[best] / 1e3, 1))
