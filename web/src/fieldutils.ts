import { row } from "./api/codec";
import type { DecodedField } from "./api/types";

export function absMax(f: DecodedField): number {
  let m = 0;
  for (let i = 0; i < f.data.length; i++) m = Math.max(m, Math.abs(f.data[i]));
  return m;
}

export function maxVal(f: DecodedField): number {
  let m = 0;
  for (let i = 0; i < f.data.length; i++) m = Math.max(m, f.data[i]);
  return m;
}

/** |a - b| on a shared grid (predictions and truth come on the same simulate grid). */
export function errorField(a: DecodedField, b: DecodedField): DecodedField {
  const n = Math.min(a.data.length, b.data.length);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.abs(a.data[i] - b.data[i]);
  return { data, nx: a.nx, nt: a.nt };
}

export { row };
