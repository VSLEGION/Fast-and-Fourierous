import { scaleLog } from "d3-scale";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { MODEL_COLORS } from "../api/types";
import type { DecodedField } from "../api/types";
import { useStore } from "../store";

const W = 300;
const H = 220;
const PAD = { l: 40, r: 10, t: 10, b: 22 };
const FNO_KMAX = 32;

function spectrum(data: Float32Array, nx: number, i: number): Float32Array {
  const u = data.subarray(i * nx, (i + 1) * nx);
  const K = Math.floor(nx / 2);
  const mag = new Float32Array(K);
  for (let k = 1; k <= K; k++) {
    let a = 0;
    let b = 0;
    for (let j = 0; j < nx; j++) {
      const ph = (-2 * Math.PI * k * j) / nx;
      a += u[j] * Math.cos(ph);
      b += u[j] * Math.sin(ph);
    }
    mag[k - 1] = Math.max(1e-9, (Math.sqrt(a * a + b * b) * 2) / nx);
  }
  return mag;
}

export function Spectrum() {
  const { truth, predictions, nt, tIndex } = useStore(
    useShallow((s) => ({
      truth: s.truth,
      predictions: s.predictions,
      nt: s.nt,
      tIndex: s.tIndex,
    })),
  );
  const i = Math.min(tIndex, nt - 1);

  const specs = useMemo(() => {
    const out: { m: string; mag: Float32Array }[] = [];
    if (truth) out.push({ m: "truth", mag: spectrum(truth.field.data, truth.field.nx, Math.min(i, truth.field.nt - 1)) });
    const fno = predictions.fno as { field: DecodedField } | undefined;
    if (fno) out.push({ m: "fno", mag: spectrum(fno.field.data, fno.field.nx, Math.min(i, fno.field.nt - 1)) });
    return out;
  }, [truth, predictions.fno, i]);

  if (!truth) return <div className="panel" style={{ padding: 12 }}><span className="microlabel">05 — Spectrum</span></div>;

  const K = specs[0].mag.length;
  const sx = scaleLog([1, K], [PAD.l, W - PAD.r]).clamp(true);
  const sy = scaleLog([1e-5, 1.5], [H - PAD.b, PAD.t]).clamp(true);
  const path = (mag: Float32Array) => {
    let d = "";
    for (let k = 1; k <= mag.length; k++) d += (k === 1 ? "M" : "L") + sx(k).toFixed(1) + " " + sy(mag[k - 1]).toFixed(1);
    return d;
  };

  return (
    <div className="panel" style={{ padding: 12 }}>
      <span className="microlabel">05 — Spectrum |û_k| (log–log) at t*</span>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block", marginTop: 6 }}>
        <rect x={sx(FNO_KMAX)} y={PAD.t} width={W - PAD.r - sx(FNO_KMAX)} height={H - PAD.b - PAD.t} fill="var(--c-fno)" opacity={0.08} />
        <text x={sx(FNO_KMAX) + 3} y={PAD.t + 10} className="microlabel" fill="var(--c-fno)">FNO k_max=32</text>
        {[1e-4, 1e-2, 1].map((v) => (
          <g key={v}>
            <line x1={PAD.l} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="var(--grid-weak)" />
            <text x={2} y={sy(v) + 3} className="num" fontSize={8} fill="var(--muted)">{v.toExponential(0)}</text>
          </g>
        ))}
        {[1, 10, 100].map((v) => (
          <text key={v} x={sx(Math.min(v, K))} y={H - 6} className="num" fontSize={8} fill="var(--muted)" textAnchor="middle">{v}</text>
        ))}
        {specs.map(({ m, mag }) => (
          <path key={m} d={path(mag)} fill="none" stroke={MODEL_COLORS[m]} strokeWidth={m === "truth" ? 1.8 : 1.3} opacity={0.9} />
        ))}
      </svg>
    </div>
  );
}
