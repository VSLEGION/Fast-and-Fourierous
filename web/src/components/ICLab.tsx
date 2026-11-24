import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { xGrid } from "../ic";
import { NU_DEFAULT, useStore } from "../store";
import { DrawCanvas } from "./DrawCanvas";
import { OODMeter } from "./OODMeter";

function U0Preview({ u0 }: { u0: Float32Array }) {
  const n = u0.length;
  const x = xGrid(n);
  let A = 0.6;
  for (let j = 0; j < n; j++) A = Math.max(A, Math.abs(u0[j]));
  const sx = scaleLinear([-1, 1], [2, 198]);
  const sy = scaleLinear([-A * 1.1, A * 1.1], [46, 2]);
  let d = "";
  for (let j = 0; j < n; j++) d += (j === 0 ? "M" : "L") + sx(x[j]).toFixed(1) + " " + sy(u0[j]).toFixed(1);
  return (
    <svg viewBox="0 0 200 48" style={{ width: "100%", height: 48, background: "var(--panel)", borderRadius: 6 }}>
      <line x1={2} y1={sy(0)} x2={198} y2={sy(0)} stroke="var(--grid-weak)" />
      <path d={d} fill="none" stroke="var(--ink)" strokeWidth={1.4} />
    </svg>
  );
}

export function ICLab() {
  const { presets, presetId, params, nu, nx, u0 } = useStore(
    useShallow((s) => ({
      presets: s.presets,
      presetId: s.presetId,
      params: s.params,
      nu: s.nu,
      nx: s.nx,
      u0: s.u0,
    })),
  );
  const setPreset = useStore((s) => s.setPreset);
  const setParam = useStore((s) => s.setParam);
  const setNu = useStore((s) => s.setNu);
  const setNx = useStore((s) => s.setNx);
  const [draw, setDraw] = useState(false);

  const preset = presets.find((p) => p.id === presetId);

  return (
    <div className="panel" style={{ padding: 14, height: "100%", overflowY: "auto" }}>
      <div className="microlabel" style={{ marginBottom: 8 }}>01 — Initial condition</div>
      <U0Preview u0={u0} />

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "12px 0" }}>
        {presets.map((p) => (
          <button
            key={p.id}
            className={p.id === presetId ? "active" : ""}
            style={{ fontSize: 11, padding: "4px 10px" }}
            onClick={() => {
              setDraw(p.id === "drawn");
              setPreset(p.id);
            }}
          >
            {p.label}
            {p.ood ? " ◇" : ""}
          </button>
        ))}
      </div>

      {preset && Object.keys(preset.params).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          {Object.entries(preset.params).map(([k, spec]) => (
            <label key={k} style={{ display: "block", marginBottom: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="microlabel">{k}</span>
                <span className="num" style={{ fontSize: 11 }}>{(params[k] ?? spec.default).toFixed(spec.step < 1 ? 2 : 0)}</span>
              </div>
              <input
                type="range"
                min={spec.min}
                max={spec.max}
                step={spec.step}
                value={params[k] ?? spec.default}
                onChange={(e) => setParam(k, +e.target.value)}
              />
            </label>
          ))}
        </div>
      )}

      <button className={draw ? "active" : ""} style={{ width: "100%", marginBottom: draw ? 8 : 12 }} onClick={() => { const d = !draw; setDraw(d); if (d) setPreset("drawn"); }}>
        {draw ? "Drawing mode" : "Draw your own"}
      </button>
      {draw && <DrawCanvas />}

      <div style={{ borderTop: "1px solid var(--hairline)", margin: "12px 0", paddingTop: 12 }}>
        <div className="microlabel" style={{ marginBottom: 6 }}>Grid N (resolution)</div>
        <div style={{ display: "flex", gap: 6 }}>
          {[256, 512, 1024].map((n) => (
            <button key={n} className={n === nx ? "active" : ""} style={{ flex: 1, fontSize: 11, padding: "4px 0" }} onClick={() => setNx(n)}>
              {n}
            </button>
          ))}
        </div>
      </div>

      <div style={{ borderTop: "1px solid var(--hairline)", paddingTop: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span className="microlabel">ν (truth only)</span>
          <span className="num" style={{ fontSize: 11 }}>{nu.toExponential(2)}</span>
        </div>
        <input type="range" min={NU_DEFAULT * 0.25} max={NU_DEFAULT * 4} step={NU_DEFAULT * 0.05} value={nu} onChange={(e) => setNu(+e.target.value)} />
        {Math.abs(nu - NU_DEFAULT) > 1e-6 && (
          <span className="pill" style={{ color: "var(--accent-b)", borderColor: "var(--accent-b)" }}>OOD · operator learned for one ν</span>
        )}
      </div>

      <div style={{ borderTop: "1px solid var(--hairline)", margin: "12px 0 0", paddingTop: 12 }}>
        <OODMeter />
      </div>
    </div>
  );
}
