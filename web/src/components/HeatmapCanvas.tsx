import { type ReactNode, useEffect, useRef } from "react";
import type { DecodedField } from "../api/types";
import { Heatmap } from "../viz/heatmap";

export function HeatmapCanvas({
  field,
  vmin,
  vmax,
  lut,
  log = false,
  exportTag,
  children,
}: {
  field: DecodedField | null;
  vmin: number;
  vmax: number;
  lut: string;
  log?: boolean;
  exportTag?: string;
  children?: ReactNode;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hmRef = useRef<Heatmap | null>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    const c = canvasRef.current;
    const hm = hmRef.current;
    if (!c || !hm || !field) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    hm.draw(field, vmin, vmax, lut, log);
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    try {
      hmRef.current = new Heatmap(canvasRef.current);
    } catch (e) {
      console.error("Heatmap init failed", e);
    }
    const ro = new ResizeObserver(() => drawRef.current());
    ro.observe(canvasRef.current);
    return () => {
      ro.disconnect();
      hmRef.current?.dispose();
      hmRef.current = null;
    };
  }, []);

  useEffect(() => {
    drawRef.current();
  }, [field, vmin, vmax, lut, log]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <canvas ref={canvasRef} data-field-export={exportTag} style={{ width: "100%", height: "100%", display: "block", borderRadius: 8 }} />
      {children}
    </div>
  );
}
