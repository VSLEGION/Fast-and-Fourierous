// Client-side initial-condition builders, mirroring shocklab/physics/ic.py so the
// heatmap and the request use the same u0 (the server recomputes truth from the
// preset + params). Held-out samples are fetched from the server instead.

export function xGrid(nx: number): Float32Array {
  const x = new Float32Array(nx);
  for (let j = 0; j < nx; j++) x[j] = -1 + (2 * j) / nx;
  return x;
}

const PI = Math.PI;

export function buildU0(presetId: string, p: Record<string, number>, nx: number): Float32Array {
  const x = xGrid(nx);
  const u = new Float32Array(nx);
  const g = (k: string, d: number) => (p[k] ?? d);
  switch (presetId) {
    case "classic":
      for (let j = 0; j < nx; j++) u[j] = -Math.sin(PI * x[j]);
      break;
    case "shifted":
      for (let j = 0; j < nx; j++) u[j] = -Math.sin(PI * (x[j] - g("s", 0.3)));
      break;
    case "boosted":
      for (let j = 0; j < nx; j++) u[j] = -Math.sin(PI * x[j]) + g("c", 0.4);
      break;
    case "linear":
      for (let j = 0; j < nx; j++) u[j] = g("eps", 0.05) * Math.sin(PI * x[j]);
      break;
    case "two_wave":
      for (let j = 0; j < nx; j++)
        u[j] = g("A", 0.6) * Math.sin(PI * x[j] + g("phi1", 0)) + g("B", 0.4) * Math.sin(2 * PI * x[j] + g("phi2", 0));
      break;
    case "n_wave": {
      let mx = 0;
      for (let j = 0; j < nx; j++) {
        let s = 0;
        for (let k = 1; k <= 16; k++) s += ((k % 2 === 1 ? 1 : -1) * Math.sin(k * PI * x[j])) / k;
        u[j] = s;
        mx = Math.max(mx, Math.abs(s));
      }
      const a = g("amp", 1.0) / (mx || 1);
      for (let j = 0; j < nx; j++) u[j] *= a;
      break;
    }
    case "gaussian": {
      const x0 = g("x0", 0),
        w = g("width", 0.15);
      let mean = 0;
      for (let j = 0; j < nx; j++) {
        let v = 0;
        for (const m of [-1, 0, 1]) v += Math.exp(-(((x[j] - x0 - 2 * m) / w) ** 2));
        u[j] = v;
        mean += v;
      }
      mean /= nx;
      for (let j = 0; j < nx; j++) u[j] -= mean;
      break;
    }
    default:
      for (let j = 0; j < nx; j++) u[j] = -Math.sin(PI * x[j]);
  }
  return u;
}

/** Project a sampled curve onto a periodic, low-pass IC (hand-drawn input, CLAUDE.md §6.3). */
export function lowpassProject(
  y: Float32Array,
  maxMode = 32,
  zeroMean = false,
  clamp = 1.2,
): Float32Array {
  const n = y.length;
  const out = new Float32Array(n);
  // naive real DFT -> keep modes <= maxMode -> inverse (n=256 => trivial cost)
  const re: number[] = [],
    im: number[] = [];
  const kMax = Math.min(maxMode, Math.floor(n / 2));
  for (let k = 0; k <= kMax; k++) {
    let a = 0,
      b = 0;
    for (let j = 0; j < n; j++) {
      const ph = (-2 * PI * k * j) / n;
      a += y[j] * Math.cos(ph);
      b += y[j] * Math.sin(ph);
    }
    re[k] = a;
    im[k] = b;
  }
  if (zeroMean) {
    re[0] = 0;
    im[0] = 0;
  }
  for (let j = 0; j < n; j++) {
    let s = re[0] / n;
    for (let k = 1; k <= kMax; k++) {
      const ph = (2 * PI * k * j) / n;
      s += (2 / n) * (re[k] * Math.cos(ph) - im[k] * Math.sin(ph));
    }
    out[j] = Math.max(-clamp, Math.min(clamp, s));
  }
  return out;
}
