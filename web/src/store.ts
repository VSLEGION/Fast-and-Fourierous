import { create } from "zustand";
import { decodeField, encodeFloat32 } from "./api/codec";
import type {
  DecodedField,
  ModelsInfo,
  OodLevel,
  Preset,
  Scorecard,
  ServerMsg,
  TruthSource,
} from "./api/types";
import { WSClient, wsUrl } from "./api/ws";
import { buildU0 } from "./ic";

export const NU_DEFAULT = 0.01 / Math.PI;
const ALL_MODELS = ["fno", "pino", "pinn"];

interface Pred {
  field: DecodedField;
  ms: number;
}
interface Truth {
  field: DecodedField;
  source: TruthSource;
  ms: number;
}
interface Training {
  task: "train_pinn" | "refine";
  iter: number;
  losses: Record<string, number>;
  preview: DecodedField | null;
  running: boolean;
  history: { iter: number; losses: Record<string, number> }[];
}

interface Store {
  connected: boolean;
  presets: Preset[];
  models: ModelsInfo;

  presetId: string;
  params: Record<string, number>;
  nu: number;
  nx: number;
  nt: number;
  u0: Float32Array;

  reqId: string;
  predictions: Record<string, Pred>;
  refined: Pred | null;
  pinnStatus: string | null;
  truth: Truth | null;
  metrics: Record<string, Scorecard>;
  ood: { level: OodLevel; reasons: string[] } | null;

  theme: "paper" | "studio";
  activeModels: Record<string, boolean>;
  tIndex: number;
  playing: boolean;
  presenter: boolean;
  storyStep: number;
  training: Training | null;

  init: () => void;
  setPreset: (id: string) => void;
  setParam: (k: string, v: number) => void;
  setNu: (v: number) => void;
  setNx: (v: number) => void;
  setDrawn: (u0: Float32Array) => void;
  loadHeldout: (index: number) => Promise<void>;
  simulate: () => void;
  trainPinn: () => void;
  refine: () => void;
  cancel: () => void;
  toggleTheme: () => void;
  toggleModel: (m: string) => void;
  setTIndex: (i: number) => void;
  togglePlay: () => void;
  setPresenter: (b: boolean) => void;
  setStory: (s: number) => void;
}

let socket: WSClient | null = null;
let simTimer: number | undefined;

export const useStore = create<Store>((set, get) => {
  const onMsg = (m: ServerMsg) => {
    const s = get();
    if ("id" in m && m.id && m.id !== s.reqId && m.type !== "progress" && m.type !== "done") {
      // allow progress/done from long tasks (different id) through; drop stale simulate msgs
      if (m.type !== "error") return;
    }
    switch (m.type) {
      case "ood":
        set({ ood: { level: m.level, reasons: m.reasons } });
        break;
      case "prediction":
        set({ predictions: { ...get().predictions, [m.model]: { field: decodeField(m.field), ms: m.ms } } });
        break;
      case "pinn_status":
        set({ pinnStatus: m.status });
        break;
      case "truth":
        set({ truth: { field: decodeField(m.field), source: m.source, ms: m.ms } });
        break;
      case "metrics":
        set({ metrics: m.per_model });
        break;
      case "progress": {
        const prev = get().training;
        set({
          training: {
            task: m.task,
            iter: m.iter,
            losses: m.losses,
            preview: decodeField(m.preview),
            running: true,
            history: [...(prev?.history ?? []), { iter: m.iter, losses: m.losses }],
          },
        });
        break;
      }
      case "done": {
        if (m.task === "train_pinn" && m.field) {
          set({
            predictions: { ...get().predictions, pinn: { field: decodeField(m.field), ms: 0 } },
            pinnStatus: null,
            training: get().training ? { ...get().training!, running: false } : null,
          });
        } else if (m.task === "refine" && m.field) {
          set({ refined: { field: decodeField(m.field), ms: 0 }, training: get().training ? { ...get().training!, running: false } : null });
        }
        break;
      }
      case "error":
        console.warn("server error:", m.error);
        if (get().training) set({ training: { ...get().training!, running: false } });
        break;
    }
  };

  const scheduleSim = () => {
    clearTimeout(simTimer);
    simTimer = setTimeout(() => get().simulate(), 40) as unknown as number;
  };

  const recomputeU0 = () => {
    const { presetId, params, nx } = get();
    if (presetId !== "drawn" && presetId !== "heldout") {
      set({ u0: buildU0(presetId, params, nx) });
    }
  };

  return {
    connected: false,
    presets: [],
    models: {},
    presetId: "classic",
    params: {},
    nu: NU_DEFAULT,
    nx: 256,
    nt: 101,
    u0: buildU0("classic", {}, 256),
    reqId: "",
    predictions: {},
    refined: null,
    pinnStatus: null,
    truth: null,
    metrics: {},
    ood: null,
    theme: "paper",
    activeModels: { truth: true, fno: true, pino: true, pinn: true },
    tIndex: 100,
    playing: false,
    presenter: false,
    storyStep: 0,
    training: null,

    init: () => {
      document.documentElement.dataset.theme = get().theme;
      fetch("/api/presets")
        .then((r) => r.json())
        .then((d) => set({ presets: d.presets }))
        .catch(() => {});
      fetch("/api/models")
        .then((r) => r.json())
        .then((d) => set({ models: d }))
        .catch(() => {});
      socket = new WSClient(
        wsUrl(),
        onMsg,
        () => {
          set({ connected: true });
          get().simulate();
        },
        () => set({ connected: false }),
      );
      socket.connect();
    },

    setPreset: (id) => {
      const preset = get().presets.find((p) => p.id === id);
      const params: Record<string, number> = {};
      if (preset) for (const [k, spec] of Object.entries(preset.params)) params[k] = spec.default;
      set({ presetId: id, params, pinnStatus: null, refined: null });
      if (id === "heldout") {
        get().loadHeldout(params.index ?? 0);
      } else if (id !== "drawn") {
        recomputeU0();
        scheduleSim();
      }
    },
    setParam: (k, v) => {
      set({ params: { ...get().params, [k]: v } });
      if (get().presetId === "heldout" && k === "index") {
        get().loadHeldout(v);
      } else {
        recomputeU0();
        scheduleSim();
      }
    },
    setNu: (v) => {
      set({ nu: v });
      scheduleSim();
    },
    setNx: (v) => {
      set({ nx: v });
      if (get().presetId === "heldout") get().loadHeldout(get().params.index ?? 0);
      else {
        recomputeU0();
        scheduleSim();
      }
    },
    setDrawn: (u0) => {
      set({ presetId: "drawn", u0, refined: null });
      scheduleSim();
    },
    loadHeldout: async (index) => {
      try {
        const r = await fetch(`/api/heldout/${index}?nx=${get().nx}`);
        const d = await r.json();
        const u0 = decodeField({ b64: d.u0.b64, nx: d.u0.n, nt: 1 }).data;
        set({ u0 });
        get().simulate();
      } catch {
        /* ignore */
      }
    },

    simulate: () => {
      if (!socket) return;
      const { u0, nu, nx, nt, presetId, params } = get();
      const reqId = crypto.randomUUID();
      set({ reqId, predictions: {}, truth: null, metrics: {}, pinnStatus: null });
      socket.send({
        type: "simulate",
        id: reqId,
        u0: { b64: encodeFloat32(u0), n: nx },
        nu,
        grid: { nx, nt },
        models: ALL_MODELS,
        preset: presetId === "drawn" || presetId === "heldout" ? presetId : presetId,
        preset_params: params,
      });
    },
    trainPinn: () => {
      if (!socket) return;
      const { u0, nx } = get();
      set({ training: { task: "train_pinn", iter: 0, losses: {}, preview: null, running: true, history: [] } });
      socket.send({ type: "train_pinn", id: crypto.randomUUID(), u0: { b64: encodeFloat32(u0), n: nx }, config: "fast" });
    },
    refine: () => {
      if (!socket) return;
      const { u0, nx, nt } = get();
      set({ training: { task: "refine", iter: 0, losses: {}, preview: null, running: true, history: [] } });
      socket.send({ type: "refine", id: crypto.randomUUID(), u0: { b64: encodeFloat32(u0), n: nx }, steps: 300, grid: { nx, nt } });
    },
    cancel: () => {
      socket?.send({ type: "cancel", id: get().reqId });
      if (get().training) set({ training: { ...get().training!, running: false } });
    },

    toggleTheme: () => {
      const theme = get().theme === "paper" ? "studio" : "paper";
      document.documentElement.dataset.theme = theme;
      set({ theme });
    },
    toggleModel: (m) => set({ activeModels: { ...get().activeModels, [m]: !get().activeModels[m] } }),
    setTIndex: (i) => set({ tIndex: i }),
    togglePlay: () => set({ playing: !get().playing }),
    setPresenter: (b) => set({ presenter: b }),
    setStory: (s) => set({ storyStep: s }),
  };
});
