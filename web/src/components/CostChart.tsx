import { scaleLinear, scaleLog } from "d3-scale";
import { MODEL_COLORS, MODEL_LABELS } from "../api/types";
import { useStore } from "../store";

const W = 300;
const H = 200;
const NMAX = 60;

function num(x: unknown, d: number): number {
  return typeof x === "number" ? x : d;
}

export function CostChart() {
  const models = useStore((s) => s.models);
  const pinnPer = num(models.pinn_fast?.metrics?.train_seconds, 64); // retrain per new IC
  const fnoOnce = num(models.fno?.metrics?.train_seconds, 1870);
  const pinoOnce = num(models.pino?.metrics?.train_seconds, 6631);
  const fnoPer = num((models.fno?.metrics?.latency_256_ms as { p50?: number })?.p50, 3) / 1000;
  const pinoPer = num((models.pino?.metrics?.latency_256_ms as { p50?: number })?.p50, 4) / 1000;

  const cum = (n: number, once: number, per: number) => once + n * per;
  const crossover = Math.round(fnoOnce / Math.max(pinnPer - fnoPer, 1e-6));

  const sx = scaleLinear([0, NMAX], [36, W - 8]);
  const sy = scaleLog([1, 1e5], [H - 20, 8]).clamp(true);
  const line = (once: number, per: number) => {
    let d = "";
    for (let n = 0; n <= NMAX; n++) d += (n === 0 ? "M" : "L") + sx(n).toFixed(1) + " " + sy(Math.max(1, cum(n, once, per))).toFixed(1);
    return d;
  };

  return (
    <div className="panel" style={{ padding: 12 }}>
      <span className="microlabel">Cost amortization · cumulative wall-clock vs queries</span>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block", marginTop: 6 }}>
        {[1, 1e1, 1e2, 1e3, 1e4].map((v) => (
          <g key={v}>
            <line x1={36} y1={sy(v)} x2={W - 8} y2={sy(v)} stroke="var(--grid-weak)" />
            <text x={2} y={sy(v) + 3} className="num" fontSize={8} fill="var(--muted)">{v >= 1000 ? v / 1000 + "k" : v}</text>
          </g>
        ))}
        {[0, 20, 40, 60].map((v) => (
          <text key={v} x={sx(v)} y={H - 4} className="num" fontSize={8} fill="var(--muted)" textAnchor="middle">{v}</text>
        ))}
        <path d={line(0, pinnPer)} fill="none" stroke={MODEL_COLORS.pinn} strokeWidth={1.6} strokeDasharray="4 3" />
        <path d={line(fnoOnce, fnoPer)} fill="none" stroke={MODEL_COLORS.fno} strokeWidth={1.6} />
        <path d={line(pinoOnce, pinoPer)} fill="none" stroke={MODEL_COLORS.pino} strokeWidth={1.6} />
        {crossover <= NMAX && (
          <>
            <line x1={sx(crossover)} y1={8} x2={sx(crossover)} y2={H - 20} stroke="var(--ink)" strokeDasharray="2 2" opacity={0.5} />
            <text x={sx(crossover) + 3} y={20} className="num" fontSize={9} fill="var(--ink)">crossover ≈ {crossover} queries</text>
          </>
        )}
      </svg>
      <div style={{ display: "flex", gap: 12, fontSize: 10 }}>
        <span className="num" style={{ color: MODEL_COLORS.pinn }}>PINN: retrain/query ({pinnPer.toFixed(0)}s)</span>
        <span className="num" style={{ color: MODEL_COLORS.fno }}>{MODEL_LABELS.fno}: once + ms</span>
      </div>
    </div>
  );
}
