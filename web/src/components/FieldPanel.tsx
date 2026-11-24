import { useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { MODEL_COLORS, MODEL_LABELS } from "../api/types";
import { absMax } from "../fieldutils";
import { useStore } from "../store";
import { HeatmapCanvas } from "./HeatmapCanvas";

function characteristics(u0: Float32Array, nx: number) {
  const dx = 2 / nx;
  let minSlope = 0;
  const slope = new Float32Array(nx);
  for (let j = 0; j < nx; j++) {
    const a = u0[(j + 1) % nx];
    const b = u0[(j - 1 + nx) % nx];
    slope[j] = (a - b) / (2 * dx);
    minSlope = Math.min(minSlope, slope[j]);
  }
  const tb = minSlope < -1e-6 ? Math.min(1, -1 / minSlope) : 1;
  const lines: [number, number][][] = [];
  const step = Math.max(1, Math.floor(nx / 26));
  for (let j = 0; j < nx; j += step) {
    const x0 = -1 + (2 * j) / nx;
    const x1 = x0 + u0[j] * tb;
    lines.push([
      [0, (1 - x0) / 2],
      [tb, (1 - x1) / 2],
    ]);
  }
  return { lines, tb };
}

function Overlay({
  showChars,
  tStar,
  hover,
}: {
  showChars: boolean;
  tStar: number;
  hover: { tf: number; yf: number } | null;
}) {
  const { u0, nx } = useStore(useShallow((s) => ({ u0: s.u0, nx: s.nx })));
  const chars = useMemo(() => characteristics(u0, nx), [u0, nx]);
  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    >
      {showChars &&
        chars.lines.map((ln, i) => (
          <polyline
            key={i}
            points={ln.map((p) => `${p[0]},${p[1]}`).join(" ")}
            fill="none"
            stroke="rgba(255,255,255,0.35)"
            strokeWidth={0.0015}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      {showChars && (
        <line x1={chars.tb} y1={0} x2={chars.tb} y2={1} stroke="#fff" strokeWidth={0.004} strokeDasharray="0.01 0.01" />
      )}
      <line x1={tStar} y1={0} x2={tStar} y2={1} stroke="var(--accent-a)" strokeWidth={0.004} opacity={0.6} />
      {hover && (
        <>
          <line x1={hover.tf} y1={0} x2={hover.tf} y2={1} stroke="var(--ink)" strokeWidth={0.003} opacity={0.5} />
          <line x1={0} y1={1 - hover.yf} x2={1} y2={1 - hover.yf} stroke="var(--ink)" strokeWidth={0.003} opacity={0.5} />
        </>
      )}
    </svg>
  );
}

export function FieldPanel() {
  const { truth, predictions, activeModels, nt, tIndex, nx } = useStore(
    useShallow((s) => ({
      truth: s.truth,
      predictions: s.predictions,
      activeModels: s.activeModels,
      nt: s.nt,
      tIndex: s.tIndex,
      nx: s.nx,
    })),
  );
  const [hover, setHover] = useState<{ tf: number; yf: number } | null>(null);

  const A = truth ? Math.max(0.6, absMax(truth.field)) : 1;
  const tStar = tIndex / (nt - 1);

  const cells = ["truth", "fno", "pino", "pinn"]
    .filter((m) => activeModels[m])
    .map((m) => {
      const field = m === "truth" ? truth?.field ?? null : predictions[m]?.field ?? null;
      const ms = m === "truth" ? truth?.ms : predictions[m]?.ms;
      return { m, field, ms };
    });

  const hoverVals = hover
    ? cells.map(({ m, field }) => {
        if (!field) return { m, v: null as number | null };
        const ti = Math.round(hover.tf * (field.nt - 1));
        const xi = Math.round(hover.yf * (field.nx - 1));
        return { m, v: field.data[ti * field.nx + xi] };
      })
    : null;

  return (
    <div className="panel" style={{ padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
        <span className="microlabel">02 — Space–time fields&nbsp;&nbsp;u(x, t)</span>
        <span className="microlabel">t → · x ↑ · n{nx}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${cells.length || 1}, 1fr)`, gap: 10 }}>
        {cells.map(({ m, field, ms }) => (
          <div key={m}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <span className="microlabel" style={{ color: MODEL_COLORS[m] }}>
                {MODEL_LABELS[m]}
              </span>
              {m === "truth" && truth && <span className="pill">{truth.source}</span>}
              {m !== "truth" && ms != null && <span className="num" style={{ fontSize: 10, color: "var(--muted)" }}>{ms.toFixed(1)} ms</span>}
            </div>
            <div
              style={{ position: "relative", aspectRatio: "1.15", overflow: "hidden", borderRadius: 8, background: "var(--panel)" }}
              onMouseMove={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setHover({ tf: (e.clientX - r.left) / r.width, yf: 1 - (e.clientY - r.top) / r.height });
              }}
              onMouseLeave={() => setHover(null)}
            >
              <HeatmapCanvas field={field} vmin={-A} vmax={A} lut="balance" exportTag={m} />
              {field && <Overlay showChars={m === "truth"} tStar={tStar} hover={hover} />}
              {!field && (
                <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
                  <span className="microlabel">{m === "pinn" ? "—" : "computing…"}</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      {hoverVals && (
        <div style={{ display: "flex", gap: 14, marginTop: 8 }}>
          {hoverVals.map(({ m, v }) => (
            <span key={m} className="num" style={{ fontSize: 11, color: MODEL_COLORS[m] }}>
              {MODEL_LABELS[m]} {v == null ? "—" : v.toFixed(3)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
