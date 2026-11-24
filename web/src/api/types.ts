// WebSocket / REST message types — mirror shocklab/server/schemas.py (keep in sync).

export interface FieldPayload {
  b64: string;
  nx: number;
  nt: number;
}

export interface DecodedField {
  data: Float32Array; // row-major [nt][nx]; u(t_i, x_j) = data[i*nx + j]
  nx: number;
  nt: number;
}

export interface Scorecard {
  rel_l2: number;
  linf: number;
  shock_dx: number;
  residual: number; // normalized by the truth's own residual
  residual_rms: number;
  mass_drift: number;
  energy_increase: number;
  energy_ok: boolean;
}

export type TruthSource = "EXACT" | "SYMMETRY" | "ANALYTIC" | "HELD-OUT" | "SOLVER";
export type OodLevel = "IN-DISTRIBUTION" | "NEAR" | "OUT-OF-DISTRIBUTION";

export type ServerMsg =
  | { type: "prediction"; id: string; model: string; field: FieldPayload; ms: number }
  | { type: "truth"; id: string; source: TruthSource; field: FieldPayload; ms: number }
  | { type: "metrics"; id: string; per_model: Record<string, Scorecard> }
  | { type: "ood"; id: string; level: OodLevel; reasons: string[] }
  | { type: "pinn_status"; id: string; status: string }
  | {
      type: "progress";
      id: string;
      task: "train_pinn" | "refine";
      iter: number;
      losses: Record<string, number>;
      preview: FieldPayload;
    }
  | { type: "done"; id: string; task?: string; field?: FieldPayload; cancelled?: boolean }
  | { type: "error"; id: string | null; error: string };

// REST
export interface ParamSpec {
  min: number;
  max: number;
  default: number;
  step: number;
}
export interface Preset {
  id: string;
  label: string;
  source: TruthSource;
  ood?: boolean;
  params: Record<string, ParamSpec>;
}
export interface ModelInfo {
  available: boolean;
  kind: string | null;
  metrics: Record<string, number | object>;
  env: Record<string, unknown>;
}
export type ModelsInfo = Record<string, ModelInfo>;

export const MODEL_COLORS: Record<string, string> = {
  truth: "var(--c-truth)",
  fno: "var(--c-fno)",
  pino: "var(--c-pino)",
  pinn: "var(--c-pinn)",
  baseline: "var(--c-baseline)",
};
export const MODEL_LABELS: Record<string, string> = {
  truth: "Truth",
  fno: "FNO",
  pino: "PINO",
  pinn: "PINN",
};
