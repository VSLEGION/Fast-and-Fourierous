import katex from "katex";
import { useShallow } from "zustand/react/shallow";
import { MODEL_COLORS, MODEL_LABELS } from "../api/types";
import { useStore } from "../store";

const EQ = katex.renderToString("u_t + u\\,u_x = \\nu\\,u_{xx}", { throwOnError: false });

export function TopBar() {
  const { connected, activeModels, theme } = useStore(
    useShallow((s) => ({
      connected: s.connected,
      activeModels: s.activeModels,
      theme: s.theme,
    })),
  );
  const toggleModel = useStore((s) => s.toggleModel);
  const toggleTheme = useStore((s) => s.toggleTheme);
  const setPresenter = useStore((s) => s.setPresenter);

  return (
    <header style={{ display: "flex", alignItems: "center", gap: 18, padding: "10px 18px", borderBottom: "1px solid var(--hairline)" }}>
      <span style={{ fontWeight: 500, letterSpacing: "-0.02em", fontSize: 18 }} className="accent-text">
        SHOCK LAB
      </span>
      <span style={{ color: "var(--muted)", fontSize: 15 }} dangerouslySetInnerHTML={{ __html: EQ }} />
      <span className="microlabel">Burgers · PINN → FNO → PINO</span>
      <div style={{ flex: 1 }} />
      <div style={{ display: "flex", gap: 6 }}>
        {["truth", "fno", "pino", "pinn"].map((m) => (
          <button
            key={m}
            onClick={() => toggleModel(m)}
            style={{
              fontSize: 11,
              padding: "3px 10px",
              borderColor: activeModels[m] ? MODEL_COLORS[m] : "var(--hairline)",
              color: activeModels[m] ? MODEL_COLORS[m] : "var(--muted)",
              opacity: activeModels[m] ? 1 : 0.5,
            }}
          >
            {MODEL_LABELS[m]}
          </button>
        ))}
      </div>
      <button onClick={toggleTheme} style={{ fontSize: 11 }}>{theme === "paper" ? "◐ Studio" : "◑ Paper"}</button>
      <button onClick={() => setPresenter(true)} style={{ fontSize: 11 }}>Present ▸</button>
      <span className="pill" style={{ color: connected ? "var(--c-pino)" : "var(--accent-a)" }}>
        {connected ? "● live" : "○ offline"}
      </span>
    </header>
  );
}
