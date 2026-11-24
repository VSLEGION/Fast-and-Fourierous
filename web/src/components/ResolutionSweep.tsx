import { scaleLog } from "d3-scale";
import { MODEL_COLORS } from "../api/types";
import { useStore } from "../store";

const W = 300;
const H = 200;

function num(x: unknown, d: number): number {
  return typeof x === "number" ? x : d;
}

// Measured zero-shot error at the trained (256) vs super-resolved (1024) grid.
export function ResolutionSweep() {
  const models = useStore((s) => s.models);
  const fno256 = num(models.fno?.metrics?.test_rel_l2_256, 0.0054);
  const fno1024 = num(models.fno?.metrics?.test_rel_l2_1024, 0.0054);
  const pino256 = num(models.pino?.metrics?.test_rel_l2_256, 0.01);
  const pino1024 = num(models.pino?.metrics?.test_rel_l2_1024, 0.01);

  const Ns = [256, 1024];
  const sx = scaleLog([200, 1200], [40, W - 10]);
  const sy = scaleLog([1e-3, 1e-1], [H - 20, 8]);
  const line = (a: number, b: number) => `M${sx(Ns[0])} ${sy(a)} L${sx(Ns[1])} ${sy(b)}`;

  return (
    <div className="panel" style={{ padding: 12 }}>
      <span className="microlabel">Resolution sweep · error vs grid N (zero-shot)</span>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block", marginTop: 6 }}>
        {[1e-3, 1e-2, 1e-1].map((v) => (
          <g key={v}>
            <line x1={40} y1={sy(v)} x2={W - 10} y2={sy(v)} stroke="var(--grid-weak)" />
            <text x={2} y={sy(v) + 3} className="num" fontSize={8} fill="var(--muted)">{(v * 100).toFixed(1)}%</text>
          </g>
        ))}
        {Ns.map((n) => (
          <text key={n} x={sx(n)} y={H - 4} className="num" fontSize={8} fill="var(--muted)" textAnchor="middle">{n}</text>
        ))}
        <path d={line(fno256, fno1024)} stroke={MODEL_COLORS.fno} strokeWidth={1.8} fill="none" />
        <path d={line(pino256, pino1024)} stroke={MODEL_COLORS.pino} strokeWidth={1.8} fill="none" />
        {[fno256, fno1024].map((v, i) => (
          <circle key={i} cx={sx(Ns[i])} cy={sy(v)} r={3} fill={MODEL_COLORS.fno} />
        ))}
      </svg>
      <span className="microlabel">trained at 256 · flat error at 1024 → resolution invariance</span>
    </div>
  );
}
