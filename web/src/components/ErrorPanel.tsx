import { useShallow } from "zustand/react/shallow";
import { MODEL_COLORS, MODEL_LABELS } from "../api/types";
import { errorField, maxVal } from "../fieldutils";
import { useStore } from "../store";
import { Colorbar } from "./Colorbar";
import { HeatmapCanvas } from "./HeatmapCanvas";

export function ErrorPanel() {
  const { truth, predictions, activeModels } = useStore(
    useShallow((s) => ({
      truth: s.truth,
      predictions: s.predictions,
      activeModels: s.activeModels,
    })),
  );
  if (!truth) return null;

  const errs = ["fno", "pino", "pinn"]
    .filter((m) => activeModels[m] && predictions[m])
    .map((m) => ({ m, field: errorField(predictions[m].field, truth.field) }));
  const vmax = Math.max(1e-3, ...errs.map((e) => maxVal(e.field)));
  const vmin = vmax * 1e-3;

  return (
    <div className="panel" style={{ padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
        <span className="microlabel">03 — Error |model − truth|&nbsp;&nbsp;(log, shared scale)</span>
        <Colorbar lut="magma" label="|err|" min={vmin.toExponential(0)} max={vmax.toExponential(1)} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${errs.length || 1}, 1fr)`, gap: 10 }}>
        {errs.map(({ m, field }) => (
          <div key={m}>
            <span className="microlabel" style={{ color: MODEL_COLORS[m] }}>{MODEL_LABELS[m]}</span>
            <div style={{ aspectRatio: "1.15", overflow: "hidden", borderRadius: 8, marginTop: 4, background: "#000" }}>
              <HeatmapCanvas field={field} vmin={vmin} vmax={vmax} lut="magma" log />
            </div>
          </div>
        ))}
        {errs.length === 0 && <span className="microlabel">waiting for predictions…</span>}
      </div>
    </div>
  );
}
