// WebGL2 space-time heatmap: a field texture (R32F) + a colormap LUT texture.
// t is horizontal, x vertical (origin lower), like the classic PINN figure.
import type { DecodedField } from "../api/types";
import { LUTS } from "./colormaps";

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uField;   // [nt rows][nx cols]
uniform sampler2D uLut;
uniform float uVmin, uVmax;
uniform int uLog;
void main() {
  // screen x -> time, screen y -> space; texture S=space(width nx), T=time(height nt)
  float val = texture(uField, vec2(vUv.y, vUv.x)).r;
  float tnorm;
  if (uLog == 1) {
    float a = abs(val);
    float lo = log(uVmin + 1e-7);
    float hi = log(uVmax + 1e-7);
    tnorm = clamp((log(a + 1e-7) - lo) / (hi - lo), 0.0, 1.0);
  } else {
    tnorm = clamp((val - uVmin) / (uVmax - uVmin), 0.0, 1.0);
  }
  frag = texture(uLut, vec2(tnorm, 0.5));
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) || "shader compile failed");
  }
  return sh;
}

export class Heatmap {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private fieldTex: WebGLTexture;
  private lutTex: Record<string, WebGLTexture> = {};
  private u: Record<string, WebGLUniformLocation | null> = {};
  private linear: boolean;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;
    gl.getExtension("EXT_color_buffer_float");
    this.linear = !!gl.getExtension("OES_texture_float_linear");

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    this.prog = gl.createProgram()!;
    gl.attachShader(this.prog, vs);
    gl.attachShader(this.prog, fs);
    gl.linkProgram(this.prog);
    gl.useProgram(this.prog);

    const quad = new Float32Array([-1, -1, 3, -1, -1, 3]);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    for (const key of ["uVmin", "uVmax", "uLog", "uField", "uLut"]) {
      this.u[key] = gl.getUniformLocation(this.prog, key);
    }

    this.fieldTex = gl.createTexture()!;
    const filt = this.linear ? gl.LINEAR : gl.NEAREST;
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    for (const [name, lut] of Object.entries(LUTS)) {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut);
      this.lutTex[name] = tex;
    }
  }

  draw(field: DecodedField, vmin: number, vmax: number, lut: string, log = false): void {
    const gl = this.gl;
    const w = gl.canvas.width;
    const h = gl.canvas.height;
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, field.nx, field.nt, 0, gl.RED, gl.FLOAT, field.data);
    gl.uniform1i(this.u.uField, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex[lut]);
    gl.uniform1i(this.u.uLut, 1);

    gl.uniform1f(this.u.uVmin, vmin);
    gl.uniform1f(this.u.uVmax, vmax);
    gl.uniform1i(this.u.uLog, log ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteTexture(this.fieldTex);
    for (const t of Object.values(this.lutTex)) gl.deleteTexture(t);
    gl.deleteProgram(this.prog);
  }
}
