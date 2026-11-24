// Export the current space-time fields view as a labeled PNG (CLAUDE.md §7, `E` key).
// WebGL canvases are created with preserveDrawingBuffer:true so drawImage reads them.

const TAGS = ["truth", "fno", "pino", "pinn"];

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function exportFieldsPNG(sourceLabel: string): void {
  const canvases = TAGS.map(
    (t) => document.querySelector(`canvas[data-field-export="${t}"]`) as HTMLCanvasElement | null,
  ).filter((c): c is HTMLCanvasElement => !!c && c.width > 0);
  if (!canvases.length) return;

  const n = canvases.length;
  const cw = 360;
  const ch = 300;
  const pad = 18;
  const header = 48;
  const footer = 30;
  const out = document.createElement("canvas");
  out.width = pad * (n + 1) + cw * n;
  out.height = header + ch + footer + pad;
  const g = out.getContext("2d")!;

  g.fillStyle = cssVar("--bg", "#F7F7F5");
  g.fillRect(0, 0, out.width, out.height);
  const ink = cssVar("--ink", "#1E1E1E");
  const muted = cssVar("--muted", "#6B6B66");

  g.fillStyle = ink;
  g.font = "600 20px 'Inter Tight', sans-serif";
  g.fillText("SHOCK LAB · space–time fields  u(x, t)", pad, 30);
  g.fillStyle = muted;
  g.font = "500 11px 'JetBrains Mono', monospace";
  g.fillText(sourceLabel, pad, 44);

  canvases.forEach((c, i) => {
    const tag = c.getAttribute("data-field-export") ?? TAGS[i];
    const x = pad + i * (cw + pad);
    g.drawImage(c, x, header, cw, ch);
    g.strokeStyle = cssVar("--hairline", "#ccc");
    g.strokeRect(x, header, cw, ch);
    g.fillStyle = ink;
    g.font = "500 12px 'JetBrains Mono', monospace";
    g.fillText(tag.toUpperCase(), x, header + ch + 18);
  });

  const a = document.createElement("a");
  a.href = out.toDataURL("image/png");
  a.download = `shocklab-${Date.now()}.png`;
  a.click();
}
