// Wave Farm Lab core: per-buoy features + gradient-boosted trees (mirrors train_perbuoy.py exactly).
(function (root) {
  'use strict';
  let M = null, NF = 40;
  let feat, thr, left, right, val, leaf, roots;

  function load(model) {
    M = model;
    let total = 0;
    for (const t of model.trees) total += t.length;
    feat = new Int16Array(total); thr = new Float64Array(total);
    left = new Int32Array(total); right = new Int32Array(total);
    val = new Float64Array(total); leaf = new Uint8Array(total);
    roots = new Int32Array(model.trees.length);
    let off = 0;
    model.trees.forEach((t, ti) => {
      roots[ti] = off;
      t.forEach((nd, k) => {
        const g = off + k;
        if (nd.length === 1) { leaf[g] = 1; val[g] = nd[0]; }
        else { feat[g] = nd[0]; thr[g] = nd[1]; left[g] = off + nd[2]; right[g] = off + nd[3]; }
      });
      off += t.length;
    });
  }

  // numpy's pairwise summation (8 accumulators) so means match np.mean bit for bit
  function npSum(a, lo, n) {
    if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += a[lo + i]; return s; }
    if (n <= 128) {
      const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
      let i = 8;
      for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
      let s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) s += a[lo + i];
      return s;
    }
    let n2 = Math.floor(n / 2); n2 -= n2 % 8;
    return npSum(a, lo, n2) + npSum(a, lo + n2, n - n2);
  }

  function features(x, y) {
    const n = x.length, K = M.K, R = M.radii, CAP = M.cap, C = M.C, S = M.S;
    const u = new Float64Array(n), v = new Float64Array(n);
    for (let i = 0; i < n; i++) { u[i] = x[i] * C + y[i] * S; v[i] = -x[i] * S + y[i] * C; }
    const um = npSum(u, 0, n) / n, vm = npSum(v, 0, n) / n;
    const F = new Float64Array(n * NF);
    const m = n - 1, di = new Float64Array(m), ui = new Float64Array(m), vi = new Float64Array(m), ang = new Float64Array(m);
    const ord = new Array(m);
    for (let i = 0; i < n; i++) {
      let q = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const du = u[j] - u[i], dv = v[j] - v[i];
        ui[q] = du; vi[q] = dv; di[q] = Math.sqrt(du * du + dv * dv);
        ang[q] = Math.abs(Math.atan2(dv, du) * (180 / Math.PI));
        ord[q] = q; q++;
      }
      ord.sort((a, b) => di[a] - di[b]);          // stable, like np.argsort(kind='stable')
      const o = i * NF;
      for (let k = 0; k < K; k++) { F[o + 2 * k] = ui[ord[k]]; F[o + 2 * k + 1] = vi[ord[k]]; }
      let col = 2 * K;
      const up = [0, 0, 0, 0], dn = [0, 0, 0, 0];
      let nnUp = Infinity, nnDn = Infinity, nnSd = Infinity, front = 0, d150 = 0, d300 = 0;
      for (let q2 = 0; q2 < m; q2++) {
        const a = ang[q2], d = di[q2];
        const isUp = a <= 30, isDn = a >= 150, isSd = a >= 60 && a <= 120;
        for (let r = 0; r < 4; r++) { if (isUp && d <= R[r]) up[r]++; if (isDn && d <= R[r]) dn[r]++; }
        if (isUp && d < nnUp) nnUp = d;
        if (isDn && d < nnDn) nnDn = d;
        if (isSd && d < nnSd) nnSd = d;
        if (ui[q2] > 0) front++;
        if (d <= 150) d150++;
        if (d <= 300) d300++;
      }
      for (let r = 0; r < 4; r++) F[o + col++] = up[r];
      for (let r = 0; r < 4; r++) F[o + col++] = dn[r];
      F[o + col++] = nnUp === Infinity ? CAP : Math.min(CAP, nnUp);
      F[o + col++] = nnDn === Infinity ? CAP : Math.min(CAP, nnDn);
      F[o + col++] = nnSd === Infinity ? CAP : Math.min(CAP, nnSd);
      F[o + col++] = front / m;
      F[o + col++] = u[i] - um; F[o + col++] = v[i] - vm;
      F[o + col++] = d150; F[o + col++] = d300;
    }
    return F;
  }

  function predictRatio(F, n) {
    const out = new Float64Array(n).fill(M.base);
    for (let i = 0; i < n; i++) {
      const o = i * NF; let s = M.base;
      for (let t = 0; t < roots.length; t++) {
        let g = roots[t];
        while (!leaf[g]) g = F[o + feat[g]] <= thr[g] ? left[g] : right[g];
        s += val[g];
      }
      out[i] = s;
    }
    return out;
  }

  function predict(x, y) {
    const F = features(x, y), r = predictRatio(F, x.length);
    const p = Array.from(r, z => z * M.p_iso);
    return { perBuoy: p, total: p.reduce((a, b) => a + b, 0), ratio: r, F };
  }

  const api = { load, features, predictRatio, predict, npSum, get pIso() { return M.p_iso; }, get C() { return M.C; }, get S() { return M.S; } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.WFM = api;
})(typeof window !== 'undefined' ? window : globalThis);
