// Reproduces the demo's "Let the model optimise" button outside the browser.
// usage: node optimise_layout.js <model.json> <demo_data.json> [seed=1]
// Hill climb on the model's own predictions from the best layout in the dataset (demo_data.layouts[0]):
// 1,500 proposed single-buoy moves, Gaussian step sigma 40 m, clamp to the 1000 x 1010 m box,
// reject moves that bring two buoys within 50 m, keep a move if the predicted farm total rises.
const fs = require('fs'), path = require('path');
const W = require(path.join(__dirname, 'wfm_core.js'));
const [modelPath, dataPath, seedArg] = process.argv.slice(2);
W.load(JSON.parse(fs.readFileSync(modelPath)));
const D = JSON.parse(fs.readFileSync(dataPath)), B = D.layouts[0];
const N = 49, clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function rng32(s) { return function () { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0; while (u === 0) u = r(); const v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const x = B.x.slice(), y = B.y.slice(), r = rng32(Number(seedArg || 1)), TOTAL = 1500;
const start = W.predict(x, y).total; let best = start, kept = 0;
const ok = (i, nx, ny) => { for (let j = 0; j < N; j++) if (j !== i && Math.hypot(nx - x[j], ny - y[j]) < 50) return false; return true; };
for (let tried = 0; tried < TOTAL; tried++) {
  const i = Math.floor(r() * N);
  const nx = clamp(x[i] + gauss(r) * 40, 0, 1000), ny = clamp(y[i] + gauss(r) * 40, 0, 1010);
  if (!ok(i, nx, ny)) continue;
  const ox = x[i], oy = y[i]; x[i] = nx; y[i] = ny;
  const t = W.predict(x, y).total;
  if (t > best) { best = t; kept++; } else { x[i] = ox; y[i] = oy; }
}
console.log(JSON.stringify({ start_kW: +(start / 1e3).toFixed(1), claim_kW: +(best / 1e3).toFixed(1), moves_kept: kept, dataset_best_kW: +(B.sim_total / 1e3).toFixed(1) }));
fs.writeFileSync('optimised_layout.json', JSON.stringify({ x, y, model_total_W: best }));
