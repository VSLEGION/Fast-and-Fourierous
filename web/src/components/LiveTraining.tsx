import { scaleLinear, scaleLog } from "d3-scale";
import { absMax } from "../fieldutils";
import { useStore } from "../store";
import { HeatmapCanvas } from "./HeatmapCanvas";

const W = 300;
const H = 150;
const LOSS_COLORS: Record<string, string> = { pde: "var(--accent-a)", ic: "var(--c-pino)", bc: "var(--c-pinn)" };

export function LiveTraining() {
  const training = useStore((s) => s.training);
  const cancel = useStore((s) => s.cancel);
  if (!training) return null;

  const hist = training.history;
  const keys = hist.length ? Object.keys(hist[hist.length - 1].losses) : [];
  const maxIter = Math.max(10, ...hist.map((h) => h.iter));
  const sx = scaleLinear([0, maxIter], [36, W - 8]);
  const sy = scaleLog([1e-6, 1], [H - 18, 8]).clamp(true);
  const A = training.preview ? Math.max(0.6, absMax(training.preview)) : 1;

  return (
    <div className="panel" style={{ padding: 12, borderColor: "var(--accent-a)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span className="microlabel" style={{ color: "var(--accent-a)" }}>
          07 — Live {training.task === "train_pinn" ? "PINN training" : "physics refinement"} · iter {training.iter}
        </span>
        {training.running && <button onClick={cancel} style={{ fontSize: 11, padding: "2px 10px" }}>Cancel</button>}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 160px", gap: 10 }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
          {[1e-5, 1e-3, 1e-1].map((v) => (
            <g key={v}>
              <line x1={36} y1={sy(v)} x2={W - 8} y2={sy(v)} stroke="var(--grid-weak)" />
              <text x={2} y={sy(v) + 3} className="num" fontSize={8} fill="var(--muted)">{v.toExponential(0)}</text>
            </g>
          ))}
          {keys.map((k) => {
            let d = "";
            hist.forEach((h, idx) => {
              const v = Math.max(1e-6, h.losses[k] ?? 1e-6);
              d += (idx === 0 ? "M" : "L") + sx(h.iter).toFixed(1) + " " + sy(v).toFixed(1);
            });
            return <path key={k} d={d} fill="none" stroke={LOSS_COLORS[k] ?? "var(--ink)"} strokeWidth={1.4} />;
          })}
        </svg>
        <div>
          <div style={{ aspectRatio: "1.3", overflow: "hidden", borderRadius: 6, background: "var(--panel)" }}>
            <HeatmapCanvas field={training.preview} vmin={-A} vmax={A} lut="balance" />
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
            {keys.map((k) => (
              <span key={k} className="num" style={{ fontSize: 9, color: LOSS_COLORS[k] ?? "var(--ink)" }}>
                {k} {(hist[hist.length - 1]?.losses[k] ?? 0).toExponential(1)}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
