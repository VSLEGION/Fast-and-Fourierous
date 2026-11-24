import { scaleLinear } from "d3-scale";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { MODEL_COLORS } from "../api/types";
import { absMax } from "../fieldutils";
import { xGrid } from "../ic";
import { useStore } from "../store";

const W = 620;
const H = 250;
const PAD = { l: 38, r: 12, t: 10, b: 24 };

function rowAt(data: Float32Array, nx: number, i: number): Float32Array {
  return data.subarray(i * nx, (i + 1) * nx);
}

function shockX(u: Float32Array, x: Float32Array): number {
  const nx = u.length;
  let best = 0;
  let bi = 0;
  for (let j = 0; j < nx; j++) {
    const d = Math.abs(u[(j + 1) % nx] - u[(j - 1 + nx) % nx]);
    if (d > best) {
      best = d;
      bi = j;
    }
  }
  return x[bi];
}

export function SnapshotPlot() {
  const { truth, predictions, activeModels, nx, nt, tIndex, playing, u0 } = useStore(
    useShallow((s) => ({
      truth: s.truth,
      predictions: s.predictions,
      activeModels: s.activeModels,
      nx: s.nx,
      nt: s.nt,
      tIndex: s.tIndex,
      playing: s.playing,
      u0: s.u0,
    })),
  );
  const setTIndex = useStore((s) => s.setTIndex);
  const togglePlay = useStore((s) => s.togglePlay);

  const tb = useMemo(() => {
    const dx = 2 / nx;
    let m = 0;
    for (let j = 0; j < nx; j++) m = Math.min(m, (u0[(j + 1) % nx] - u0[(j - 1 + nx) % nx]) / (2 * dx));
    return m < -1e-6 ? Math.min(1, -1 / m) : 1;
  }, [u0, nx]);

  if (!truth) return <div className="panel" style={{ padding: 12 }}><span className="microlabel">04 — Snapshot</span></div>;

  const A = Math.max(0.6, absMax(truth.field));
  const i = Math.min(tIndex, nt - 1);
  const tStar = i / (nt - 1);
  const series = ["truth", "fno", "pino", "pinn"]
    .filter((m) => activeModels[m] && (m === "truth" ? truth : predictions[m]))
    .map((m) => {
      const f = m === "truth" ? truth.field : predictions[m].field;
      return { m, u: rowAt(f.data, f.nx, Math.min(i, f.nt - 1)), nx: f.nx };
    });

  const sx = scaleLinear([-1, 1], [PAD.l, W - PAD.r]);
  const sy = scaleLinear([-A * 1.05, A * 1.05], [H - PAD.b, PAD.t]);
  const xs = shockX(rowAt(truth.field.data, truth.field.nx, Math.min(i, truth.field.nt - 1)), xGrid(truth.field.nx));

  const path = (u: Float32Array, n: number, xlo = -1, xhi = 1) => {
    const xx = xGrid(n);
    let d = "";
    for (let j = 0; j < n; j++) {
      if (xx[j] < xlo || xx[j] > xhi) continue;
      d += (d === "" ? "M" : "L") + sx(xx[j]).toFixed(1) + " " + sy(u[j]).toFixed(1);
    }
    return d;
  };
  // Shock lens: zoomed inset around xs
  const lensW = 150, lensH = 96, lx0 = xs - 0.09, lx1 = xs + 0.09;
  const lsx = scaleLinear([lx0, lx1], [4, lensW - 4]);
  const lsy = scaleLinear([-A * 1.05, A * 1.05], [lensH - 4, 4]);
  const lpath = (u: Float32Array, n: number) => {
    const xx = xGrid(n);
    let d = "";
    for (let j = 0; j < n; j++) {
      if (xx[j] < lx0 || xx[j] > lx1) continue;
      d += (d === "" ? "M" : "L") + lsx(xx[j]).toFixed(1) + " " + lsy(u[j]).toFixed(1);
    }
    return d;
  };

  return (
    <div className="panel" style={{ padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <span className="microlabel">04 — Snapshot u(x, t*)&nbsp;&nbsp;+ shock lens</span>
        <span className="num" style={{ fontSize: 11, color: "var(--muted)" }}>t* = {tStar.toFixed(2)}</span>
      </div>
      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
          {sy.ticks(4).map((v) => (
            <g key={v}>
              <line x1={PAD.l} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="var(--grid-weak)" />
              <text x={4} y={sy(v) + 3} className="num" fontSize={9} fill="var(--muted)">{v.toFixed(1)}</text>
            </g>
          ))}
          {sx.ticks(5).map((v) => (
            <text key={v} x={sx(v)} y={H - 8} className="num" fontSize={9} fill="var(--muted)" textAnchor="middle">{v}</text>
          ))}
          <rect x={sx(lx0)} y={PAD.t} width={sx(lx1) - sx(lx0)} height={H - PAD.b - PAD.t} fill="var(--accent-a)" opacity={0.06} />
          {series.map(({ m, u, nx }) => (
            <path key={m} d={path(u, nx)} fill="none" stroke={MODEL_COLORS[m]} strokeWidth={m === "truth" ? 2 : 1.4} opacity={m === "truth" ? 0.9 : 1} />
          ))}
        </svg>
        <div style={{ position: "absolute", top: 8, right: 14, width: lensW, height: lensH, background: "var(--surface)", border: "1px solid var(--hairline)", borderRadius: 8 }}>
          <svg viewBox={`0 0 ${lensW} ${lensH}`} style={{ width: "100%", height: "100%" }}>
            {series.map(({ m, u, nx }) => (
              <path key={m} d={lpath(u, nx)} fill="none" stroke={MODEL_COLORS[m]} strokeWidth={m === "truth" ? 2 : 1.3} />
            ))}
          </svg>
          <span className="microlabel" style={{ position: "absolute", top: 3, left: 6 }}>shock lens x≈{xs.toFixed(2)}</span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
        <button onClick={togglePlay} style={{ padding: "4px 12px" }}>{playing ? "❚❚" : "▶"}</button>
        <input type="range" min={0} max={nt - 1} value={i} onChange={(e) => setTIndex(+e.target.value)} style={{ flex: 1 }} />
        <span className="num" style={{ fontSize: 10, color: "var(--accent-b)" }}>t_b≈{tb.toFixed(2)}</span>
      </div>
    </div>
  );
}
