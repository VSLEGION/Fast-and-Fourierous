// Base64 <-> Float32Array (little-endian). x86 is little-endian so a Float32Array
// view over the decoded bytes reads the server's LE floats directly.
import type { DecodedField, FieldPayload } from "./types";

export function decodeFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function decodeField(f: FieldPayload): DecodedField {
  return { data: decodeFloat32(f.b64), nx: f.nx, nt: f.nt };
}

export function encodeFloat32(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

/** A single-time row u(t_i, ·) from a row-major [nt][nx] field. */
export function row(field: DecodedField, i: number): Float32Array {
  return field.data.subarray(i * field.nx, (i + 1) * field.nx);
}
