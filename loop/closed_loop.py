# Closed loop: the per-buoy model proposes layouts, the simulator scores them, the scored layouts are added to the
# training data and the model is retrained. Plus two model-free controls with the same simulation budget.
# Rebuilt 25 Nov 2026 (the October code and layouts were lost); same design as results.md §11:
#   7 rounds x 6 proposals = 42 simulations per run, interaction cut-off kR 100, compared with the dataset's best
#   layout re-simulated, and with a model that is never retrained.
#
# Not yet run on the real simulator. With SIM_STUB=1 the controls ran end to end; loop mode has not finished a test run. Each 49-buoy layout takes ~30 min in the
# original Octave simulator, so a full run (42 simulations per run) needs many parallel workers (WORKERS=n).
# usage: python3 closed_loop.py <WEC_Perth_49.csv> <simulator dir> <climate_Perth.mat> <out dir> <mode> <seed> [rounds] [per_round]
#   mode: loop | random | rules
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))
import numpy as np
from common import load_unique, feats, feats_all, model

CSV, SIM, CLIM, OUT, MODE, SEED = sys.argv[1:7]
SEED = int(SEED)
ROUNDS = int(sys.argv[7]) if len(sys.argv) > 7 else 7
PER = int(sys.argv[8]) if len(sys.argv) > 8 else 6
KR = 100; K_PTO, D_PTO = 407400, 96500
P_ISO = 96374.0
BOX = (0.0, 1000.0, 0.0, 1010.0); MIN_GAP = 50.0
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
tag = f'{MODE}_s{SEED}'
rng = np.random.default_rng(SEED)
t0 = time.time()
log = open(f'{OUT}/{tag}.log', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


def simulate(layouts):
    """layouts: list of (x, y) -> list of (total W, per-buoy W array)."""
    if os.environ.get('SIM_STUB'):   # test mode only: a fake simulator (dataset-trained model + noise), to check the code paths
        global _stub
        if '_stub' not in globals():
            _stub = model(100, seed=7).fit(feats_all(X[:3000], Y[:3000]), (P[:3000] / P_ISO).ravel())
        out = []
        for x, y in layouts:
            pb = P_ISO * _stub.predict(feats(x, y)) * (1 + 0.002 * rng.standard_normal(len(x)))
            out.append((pb.sum(), pb))
        return out
    # split across WORKERS parallel Octave processes (each layout takes minutes)
    W = max(1, min(int(os.environ.get('WORKERS', '6')), len(layouts)))
    chunks = [list(range(len(layouts)))[w::W] for w in range(W)]
    procs = []
    for w, idx in enumerate(chunks):
        f_in, f_out = f'{OUT}/{tag}_in{w}.csv', f'{OUT}/{tag}_out{w}.csv'
        np.savetxt(f_in, np.array([np.r_[layouts[i][0], layouts[i][1]] for i in idx]), delimiter=',', fmt='%.4f')
        if os.path.exists(f_out): os.remove(f_out)
        procs.append((idx, f_out, subprocess.Popen(['octave', '-q', '--no-gui', os.path.join(HERE, '..', 'simulator', 'run_batch.m'), SIM, CLIM, f_in,
                                                     str(K_PTO), str(D_PTO), str(KR), f_out], stdout=subprocess.DEVNULL)))
    out = [None] * len(layouts)
    for idx, f_out, p in procs:
        if p.wait() != 0: raise RuntimeError(f'octave failed ({f_out})')
        R = np.atleast_2d(np.loadtxt(f_out, delimiter=','))
        for i, r in zip(idx, R): out[i] = (r[0], r[3:])
    return out


def valid(x, y):
    if x.min() < BOX[0] or x.max() > BOX[1] or y.min() < BOX[2] or y.max() > BOX[3]: return False
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]); np.fill_diagonal(d, np.inf)
    return d.min() >= MIN_GAP


def ok_at(x, y, i, nx, ny):
    if not (BOX[0] <= nx <= BOX[1] and BOX[2] <= ny <= BOX[3]): return False
    d = np.hypot(np.delete(x, i) - nx, np.delete(y, i) - ny)
    return d.min() >= MIN_GAP


def predict(m, x, y):
    return P_ISO * m.predict(feats(x, y)).sum()


def hill_climb(m, x, y, steps, rng):
    """Move one buoy at a time to a better spot according to the model (local moves and jumps)."""
    x, y = x.copy(), y.copy(); best = predict(m, x, y)
    for _ in range(steps):
        i = rng.integers(len(x))
        if rng.random() < 0.5:
            nx, ny = x[i] + rng.normal(0, 40), y[i] + rng.normal(0, 40)
        else:
            nx, ny = rng.uniform(BOX[0], BOX[1]), rng.uniform(BOX[2], BOX[3])
        if not ok_at(x, y, i, nx, ny): continue
        ox, oy = x[i], y[i]; x[i], y[i] = nx, ny
        p = predict(m, x, y)
        if p > best: best = p
        else: x[i], y[i] = ox, oy
    return x, y, best


# ---- data
X, Y, P, T = load_unique(CSV)
nu, N = X.shape
b = int(np.argmax(T))
say(f'[{tag}] {nu} unique layouts; dataset best {T[b]:,.0f} W (row {b}); rounds {ROUNDS} x {PER}')

# reference: dataset's best layout re-simulated (same settings), cached across runs
ref_f = f'{OUT}/reference_best.json'
if os.path.exists(ref_f):
    ref = json.load(open(ref_f))
else:
    tot, pb = simulate([(X[b], Y[b])])[0]
    ref = {'dataset_W': float(T[b]), 'simulated_W': float(tot), 'kR': KR}
    json.dump(ref, open(ref_f, 'w'), indent=1)
say(f'reference: dataset best re-simulated {ref["simulated_W"]:,.0f} W (dataset {ref["dataset_W"]:,.0f})')

record = {'mode': MODE, 'seed': SEED, 'reference': ref, 'rounds': [], 'simulated': []}
sim_X, sim_Y, sim_PB, sim_T = [], [], [], []

if MODE == 'loop':
    F0 = feats_all(X, Y); r0 = (P / P_ISO).ravel()
    say(f'features {F0.shape} ({time.time()-t0:.0f}s)')
    frozen = model(300).fit(F0, r0)          # never retrained
    current = frozen
    W_NEW = 20.0                             # weight of each simulated buoy row vs a dataset row
    for rd in range(ROUNDS):
        # starting points: best known layouts (dataset + simulated so far), ranked by their known power
        pool = [(T[k], X[k], Y[k]) for k in np.argsort(-T)[:30]] + list(zip(sim_T, sim_X, sim_Y))
        pool.sort(key=lambda z: -z[0])
        props = []
        for j in range(PER):
            _, sx, sy = pool[min(j, len(pool) - 1)] if j < 3 else pool[rng.integers(min(10, len(pool)))]
            px, py, pp = hill_climb(current, sx, sy, 1500, rng)
            props.append((px, py, pp, predict(frozen, px, py)))
        res = simulate([(p[0], p[1]) for p in props])
        for (px, py, pp, pf), (tot, pb) in zip(props, res):
            sim_X.append(px); sim_Y.append(py); sim_PB.append(pb); sim_T.append(tot)
            record['simulated'].append({'round': rd + 1, 'sim_W': float(tot), 'pred_retrained_W': float(pp), 'pred_frozen_W': float(pf),
                                        'x': px.round(2).tolist(), 'y': py.round(2).tolist()})
        rows = record['simulated'][-PER:]
        mae_r = np.mean([abs(z['pred_retrained_W'] - z['sim_W']) for z in rows]) / 1e3
        mae_f = np.mean([abs(z['pred_frozen_W'] - z['sim_W']) for z in rows]) / 1e3
        opt = np.mean([z['pred_retrained_W'] - z['sim_W'] for z in rows]) / 1e3
        best = max(sim_T)
        record['rounds'].append({'round': rd + 1, 'best_so_far_W': float(best), 'mae_retrained_kW': round(mae_r, 1),
                                 'mae_never_retrained_kW': round(mae_f, 1), 'mean_optimism_kW': round(opt, 1)})
        say(f'round {rd+1}: best so far {best:,.0f} W ({best-ref["simulated_W"]:+,.0f} vs reference); '
            f'MAE retrained {mae_r:.1f} kW, never retrained {mae_f:.1f} kW, optimism {opt:+.1f} kW ({time.time()-t0:.0f}s)')
        # retrain with the simulated layouts added
        Fn = np.concatenate([feats(a, c) for a, c in zip(sim_X, sim_Y)]); rn = (np.array(sim_PB) / P_ISO).ravel()
        current = model(300).fit(np.r_[F0, Fn], np.r_[r0, rn], sample_weight=np.r_[np.ones(len(r0)), W_NEW * np.ones(len(rn))])

else:
    # model-free controls, same budget, greedy from the dataset's best layout
    cx, cy, cbest = X[b].copy(), Y[b].copy(), ref['simulated_W']
    th = np.deg2rad(55.3)
    for rd in range(ROUNDS):
        props = []
        for j in range(PER):
            x, y = cx.copy(), cy.copy()
            if MODE == 'random':          # move 1-3 random buoys to random valid spots
                for _ in range(rng.integers(1, 4)):
                    i = rng.integers(N)
                    for _ in range(200):
                        nx, ny = rng.uniform(BOX[0], BOX[1]), rng.uniform(BOX[2], BOX[3])
                        if ok_at(x, y, i, nx, ny): x[i], y[i] = nx, ny; break
            else:                         # geometry rule: move a back-row buoy to the open spot with most up-wave clearance
                u = x * np.cos(th) + y * np.sin(th)
                i = int(np.argsort(u)[rng.integers(5)])
                cand = []
                for _ in range(400):
                    nx, ny = rng.uniform(BOX[0], BOX[1]), rng.uniform(BOX[2], BOX[3])
                    if not ok_at(x, y, i, nx, ny): continue
                    du = (np.delete(x, i) - nx) * np.cos(th) + (np.delete(y, i) - ny) * np.sin(th)
                    dv = -(np.delete(x, i) - nx) * np.sin(th) + (np.delete(y, i) - ny) * np.cos(th)
                    upw = (du > 0) & (np.abs(np.degrees(np.arctan2(dv, du))) <= 30)
                    clear = np.hypot(du, dv)[upw].min() if upw.any() else 2000.0
                    cand.append((clear, nx, ny))
                cand.sort(key=lambda z: -z[0])
                _, x[i], y[i] = cand[rng.integers(min(5, len(cand)))]
            props.append((x, y))
        res = simulate(props)
        for (x, y), (tot, pb) in zip(props, res):
            sim_T.append(tot)
            record['simulated'].append({'round': rd + 1, 'sim_W': float(tot), 'x': x.round(2).tolist(), 'y': y.round(2).tolist()})
            if tot > cbest: cbest, cx, cy = tot, x, y
        record['rounds'].append({'round': rd + 1, 'best_so_far_W': float(max(cbest, ref['simulated_W']))})
        say(f'round {rd+1}: best so far {cbest:,.0f} W ({cbest-ref["simulated_W"]:+,.0f} vs reference) ({time.time()-t0:.0f}s)')

best = max(sim_T)
record['summary'] = {'best_W': float(best), 'gain_vs_reference_W': float(best - ref['simulated_W']),
                     'gain_pct': round(float(100 * (best / ref['simulated_W'] - 1)), 3),
                     'n_beat_reference': int(sum(t > ref['simulated_W'] for t in sim_T)), 'n_simulated': len(sim_T),
                     'seconds': round(time.time() - t0)}
json.dump(record, open(f'{OUT}/{tag}.json', 'w'))
say(f'[{tag}] done: {record["summary"]}')
