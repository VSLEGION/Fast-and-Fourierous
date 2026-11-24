import { useStore } from "../store";

const COLOR: Record<string, string> = {
  "IN-DISTRIBUTION": "var(--c-pino)",
  NEAR: "var(--accent-a)",
  "OUT-OF-DISTRIBUTION": "var(--accent-b)",
};

export function OODMeter() {
  const ood = useStore((s) => s.ood);
  if (!ood) return null;
  const c = COLOR[ood.level] ?? "var(--muted)";
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className="microlabel">Distribution</span>
        <span className="pill" style={{ color: c, borderColor: c }}>{ood.level}</span>
      </div>
      <ul style={{ margin: "6px 0 0", paddingLeft: 16 }}>
        {ood.reasons.map((r, i) => (
          <li key={i} style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>{r}</li>
        ))}
      </ul>
    </div>
  );
}
