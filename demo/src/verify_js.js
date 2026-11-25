// usage: node verify_js.js <demo_dir> <out_dir>
const fs = require('fs');
const [demo, out] = process.argv.slice(2);
const W = require(demo + '/wfm_core.js');
W.load(JSON.parse(fs.readFileSync(demo + '/model.json')));
const V = JSON.parse(fs.readFileSync(out + '/verify_sample.json'));
// features of the best layout
const F = W.features(V.x[0], V.y[0]);
let fmax = 0;
for (let i = 0; i < 49; i++) for (let c = 0; c < 40; c++) fmax = Math.max(fmax, Math.abs(F[i * 40 + c] - V.feats0[i][c]));
console.log('features max abs diff (best layout):', fmax);
let pmax = 0, tmax = 0, t0 = Date.now();
for (let k = 0; k < V.x.length; k++) {
  const r = W.predict(V.x[k], V.y[k]);
  let tp = 0;
  for (let i = 0; i < 49; i++) { pmax = Math.max(pmax, Math.abs(r.ratio[i] - V.pred[k][i])); tp += V.pred[k][i] * 96374; }
  tmax = Math.max(tmax, Math.abs(r.total - tp));
}
const dt = (Date.now() - t0) / V.x.length;
console.log('layouts', V.x.length, 'per-buoy ratio max abs diff', pmax, 'farm total max abs diff (W)', tmax.toFixed(6), 'ms/layout', dt.toFixed(2));
