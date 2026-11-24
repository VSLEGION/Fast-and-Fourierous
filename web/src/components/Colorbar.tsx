import { lutColor } from "../viz/colormaps";

export function Colorbar({
  lut,
  label,
  min,
  max,
}: {
  lut: string;
  label: string;
  min: string;
  max: string;
}) {
  const stops = Array.from({ length: 12 }, (_, i) => lutColor(lut, i / 11)).join(", ");
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span className="num" style={{ fontSize: 10, color: "var(--muted)" }}>{min}</span>
      <div style={{ width: 120, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${stops})`, border: "1px solid var(--hairline)" }} />
      <span className="num" style={{ fontSize: 10, color: "var(--muted)" }}>{max}</span>
      <span className="microlabel" style={{ marginLeft: 4 }}>{label}</span>
    </div>
  );
}
