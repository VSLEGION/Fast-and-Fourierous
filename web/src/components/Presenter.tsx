import { useEffect } from "react";
import { useStore } from "../store";

type Step = { cap: string; run: (s: ReturnType<typeof useStore.getState>) => void };

export const STORY: Step[] = [
  { cap: "The classic shock  u₀ = −sin(πx). All three models predict against the exact answer.", run: (s) => s.setPreset("classic") },
  { cap: "The PINN learned ONE solution from physics alone — slow to train, but razor-sharp at the shock.", run: () => {} },
  { cap: "The FNO maps any IC → full evolution in ~3 ms. One model, every initial condition.", run: () => {} },
  { cap: "Gaussian pulse: a new IC the FNO has never seen, solved instantly.", run: (s) => s.setPreset("gaussian") },
  { cap: "Shift the classic case: the FNO is EXACTLY translation-equivariant (no x-channel).", run: (s) => s.setPreset("shifted") },
  { cap: "Boost it (nonzero mean) — out of distribution. The operator struggles here.", run: (s) => s.setPreset("boosted") },
  { cap: "Refine with physics: no data, just the PDE residual — the shock snaps toward the truth.", run: (s) => { s.setPreset("boosted"); setTimeout(() => s.refine(), 500); } },
  { cap: "Train once, query forever: the cost crossover vs the retrain-every-time PINN.", run: (s) => s.setPreset("classic") },
];

export function Presenter() {
  const presenter = useStore((s) => s.presenter);
  const step = useStore((s) => s.storyStep);
  const setStory = useStore((s) => s.setStory);
  const setPresenter = useStore((s) => s.setPresenter);

  useEffect(() => {
    if (presenter) STORY[step]?.run(useStore.getState());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presenter, step]);

  if (!presenter) return null;
  return (
    <div style={{ position: "fixed", left: 0, right: 0, bottom: 0, padding: "18px 28px", background: "linear-gradient(0deg, var(--bg), transparent)", display: "flex", alignItems: "center", gap: 20, zIndex: 50 }}>
      <span className="microlabel">Story {step + 1}/{STORY.length}</span>
      <p style={{ flex: 1, fontSize: 22, letterSpacing: "-0.01em", margin: 0 }}>{STORY[step].cap}</p>
      <button onClick={() => setStory(Math.max(0, step - 1))} disabled={step === 0}>←</button>
      <button className="active" onClick={() => setStory(Math.min(STORY.length - 1, step + 1))} disabled={step === STORY.length - 1}>→</button>
      <button onClick={() => setPresenter(false)}>Esc</button>
    </div>
  );
}
