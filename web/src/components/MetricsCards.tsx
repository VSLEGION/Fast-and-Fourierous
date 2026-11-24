import { useShallow } from "zustand/react/shallow";
import { MODEL_COLORS, MODEL_LABELS } from "../api/types";
import { useStore } from "../store";

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 0" }}>
      <span className="microlabel">{label}</span>
      <span className="num" style={{ fontSize: 12, color: color ?? "var(--ink)" }}>{value}</span>
    </div>
  );
}

function trainCost(models: ReturnType<typeof useStore.getState>["models"], m: string): string {
  const key = m === "pinn" ? "pinn_classic" : m;
  const s = models[key]?.metrics?.train_seconds as number | undefined;
  if (s == null) return "—";
  return s >= 60 ? `${(s / 60).toFixed(0)} min` : `${s.toFixed(0)} s`;
}

export function MetricsCards() {
  const { metrics, predictions, models, pinnStatus, training } = useStore(
    useShallow((s) => ({
      metrics: s.metrics,
      predictions: s.predictions,
      models: s.models,
      pinnStatus: s.pinnStatus,
      training: s.training,
    })),
  );
  const trainPinn = useStore((s) => s.trainPinn);
  const refine = useStore((s) => s.refine);

  return (
    <div className="panel" style={{ padding: 12 }}>
      <span className="microlabel">06 — Metrics · physics · speed</span>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginTop: 8 }}>
        {["fno", "pino", "pinn"].map((m) => {
          const sc = metrics[m];
          const pred = predictions[m];
          const isPinnRetrain = m === "pinn" && (pinnStatus === "RETRAIN_REQUIRED" || !pred);
          return (
            <div key={m} className="panel-warm" style={{ padding: 10, borderRadius: 10, borderTop: `2px solid ${MODEL_COLORS[m]}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ color: MODEL_COLORS[m], fontWeight: 500 }}>{MODEL_LABELS[m]}</span>
                <span className="num" style={{ fontSize: 11, color: "var(--muted)" }}>{pred ? `${pred.ms.toFixed(1)} ms` : ""}</span>
              </div>
              {isPinnRetrain ? (
                <div>
                  <div className="pill" style={{ color: "var(--accent-a)", marginBottom: 8 }}>RETRAIN REQUIRED</div>
                  <p style={{ fontSize: 11, color: "var(--muted)", margin: "0 0 8px" }}>
                    A PINN learns one solution. Train it on this IC (~60 s, live).
                  </p>
                  <button className="active" style={{ width: "100%" }} disabled={training?.running} onClick={trainPinn}>
                    Train PINN on this IC
                  </button>
                </div>
              ) : sc ? (
                <>
                  <Row label="rel L2" value={(sc.rel_l2 * 100).toFixed(2) + "%"} />
                  <Row label="L∞" value={sc.linf.toFixed(3)} />
                  <Row label="shock Δx" value={sc.shock_dx.toExponential(1)} />
                  <Row label="residual" value={"×" + sc.residual.toFixed(1)} color={sc.residual < 2 ? "var(--c-pino)" : "var(--ink)"} />
                  <Row label="mass drift" value={sc.mass_drift.toExponential(1)} />
                  <Row label="energy" value={sc.energy_ok ? "✓ monotone" : "⚠ rises"} color={sc.energy_ok ? "var(--c-pino)" : "var(--accent-a)"} />
                  <Row label="train cost" value={trainCost(models, m)} color="var(--muted)" />
                  {m === "pino" && (
                    <button style={{ width: "100%", marginTop: 8 }} disabled={training?.running} onClick={refine}>
                      Refine with physics
                    </button>
                  )}
                </>
              ) : (
                <span className="microlabel">computing…</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
