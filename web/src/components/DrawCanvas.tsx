import { useEffect, useRef } from "react";
import { lowpassProject } from "../ic";
import { useStore } from "../store";

const CW = 244;
const CH = 110;
const YMAX = 1.3;

export function DrawCanvas() {
  const nx = useStore((s) => s.nx);
  const setDrawn = useStore((s) => s.setDrawn);
  const u0 = useStore((s) => s.u0);
  const presetId = useStore((s) => s.presetId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vals = useRef<Float32Array>(new Float32Array(nx));
  const drawing = useRef(false);

  // seed the scratch buffer from the current IC so drawing starts from it
  useEffect(() => {
    const v = new Float32Array(nx);
    for (let j = 0; j < nx; j++) v[j] = u0[Math.min(j, u0.length - 1)] ?? 0;
    vals.current = v;
    render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nx, presetId]);

  const render = () => {
    const c = canvasRef.current;
    if (!c) return;
    const g = c.getContext("2d")!;
    const style = getComputedStyle(document.documentElement);
    g.clearRect(0, 0, CW, CH);
    g.strokeStyle = style.getPropertyValue("--grid-weak") || "#eee";
    g.beginPath();
    g.moveTo(0, CH / 2);
    g.lineTo(CW, CH / 2);
    g.stroke();
    g.strokeStyle = style.getPropertyValue("--accent-a") || "#f70";
    g.lineWidth = 1.6;
    g.beginPath();
    const v = vals.current;
    for (let j = 0; j < nx; j++) {
      const x = (j / (nx - 1)) * CW;
      const y = CH / 2 - (v[j] / YMAX) * (CH / 2);
      j === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    }
    g.stroke();
  };

  const paint = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const r = canvasRef.current!.getBoundingClientRect();
    const px = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const val = (CH / 2 - ((e.clientY - r.top) / r.height) * CH) / (CH / 2) * YMAX;
    const j = Math.round(px * (nx - 1));
    for (let k = Math.max(0, j - 2); k <= Math.min(nx - 1, j + 2); k++) vals.current[k] = val;
    render();
  };

  const commit = () => {
    drawing.current = false;
    setDrawn(lowpassProject(vals.current, 32, false, 1.2));
  };

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={CW}
        height={CH}
        style={{ width: "100%", height: CH, border: "1px solid var(--hairline)", borderRadius: 8, touchAction: "none", cursor: "crosshair" }}
        onPointerDown={(e) => {
          drawing.current = true;
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          paint(e);
        }}
        onPointerMove={paint}
        onPointerUp={commit}
      />
      <span className="microlabel">draw a curve → projected to periodic, low-pass (modes ≤ 32)</span>
    </div>
  );
}
