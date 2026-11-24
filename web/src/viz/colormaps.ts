// Perceptual colormap LUTs exported to 256-entry RGBA tables (CLAUDE.md §7):
//   - `balance`/RdBu_r: diverging, centered on 0, for the field u
//   - `magma`: sequential, for errors (shared log scale across models)

type Stop = [number, [number, number, number]];

// RdBu_r: blue (negative) -> white (0) -> red (positive)
const RDBU_R: Stop[] = [
  [0.0, [5, 48, 97]],
  [0.25, [67, 147, 195]],
  [0.5, [247, 247, 247]],
  [0.75, [214, 96, 77]],
  [1.0, [103, 0, 31]],
];

// magma-like sequential
const MAGMA: Stop[] = [
  [0.0, [0, 0, 4]],
  [0.25, [81, 18, 124]],
  [0.5, [183, 55, 121]],
  [0.75, [252, 137, 97]],
  [1.0, [252, 253, 191]],
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function buildLUT(stops: Stop[]): Uint8Array {
  const n = 256;
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    let k = 0;
    while (k < stops.length - 2 && x > stops[k + 1][0]) k++;
    const [x0, c0] = stops[k];
    const [x1, c1] = stops[k + 1];
    const t = (x - x0) / (x1 - x0 || 1);
    out[i * 4] = Math.round(lerp(c0[0], c1[0], t));
    out[i * 4 + 1] = Math.round(lerp(c0[1], c1[1], t));
    out[i * 4 + 2] = Math.round(lerp(c0[2], c1[2], t));
    out[i * 4 + 3] = 255;
  }
  return out;
}

export const LUTS: Record<string, Uint8Array> = {
  balance: buildLUT(RDBU_R),
  magma: buildLUT(MAGMA),
};

/** CSS color string from a LUT at a normalized position (for legends/colorbars). */
export function lutColor(name: string, t: number): string {
  const lut = LUTS[name];
  const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 4;
  return `rgb(${lut[i]}, ${lut[i + 1]}, ${lut[i + 2]})`;
}
