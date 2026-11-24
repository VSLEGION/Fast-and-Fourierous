import { useEffect } from "react";
import { CostChart } from "./components/CostChart";
import { ErrorPanel } from "./components/ErrorPanel";
import { FieldPanel } from "./components/FieldPanel";
import { ICLab } from "./components/ICLab";
import { LiveTraining } from "./components/LiveTraining";
import { MetricsCards } from "./components/MetricsCards";
import { Presenter, STORY } from "./components/Presenter";
import { ResolutionSweep } from "./components/ResolutionSweep";
import { SnapshotPlot } from "./components/SnapshotPlot";
import { Spectrum } from "./components/Spectrum";
import { TopBar } from "./components/TopBar";
import { useStore } from "./store";
import { exportFieldsPNG } from "./viz/export";

export function App() {
  const init = useStore((s) => s.init);
  const playing = useStore((s) => s.playing);
  const nt = useStore((s) => s.nt);
  const training = useStore((s) => s.training);
  const presenter = useStore((s) => s.presenter);

  useEffect(() => {
    init();
  }, [init]);

  // time scrubber playback
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const s = useStore.getState();
      s.setTIndex((s.tIndex + 1) % nt);
    }, 55);
    return () => clearInterval(id);
  }, [playing, nt]);

  // keyboard shortcuts (CLAUDE.md §7)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = useStore.getState();
      if (e.target instanceof HTMLInputElement) return;
      const k = e.key.toLowerCase();
      if (e.key === " ") {
        e.preventDefault();
        s.togglePlay();
      } else if (k === "r") s.refine();
      else if (k === "t") s.trainPinn();
      else if (k === "d") s.toggleTheme();
      else if (k === "e") exportFieldsPNG(`${s.truth?.source ?? "—"} · preset ${s.presetId} · N${s.nx}`);
      else if (e.key === "Escape") s.setPresenter(false);
      else if (s.presenter && e.key === "ArrowRight") s.setStory(Math.min(STORY.length - 1, s.storyStep + 1));
      else if (s.presenter && e.key === "ArrowLeft") s.setStory(Math.max(0, s.storyStep - 1));
      else if (/^[1-9]$/.test(e.key)) {
        const p = s.presets[+e.key - 1];
        if (p) s.setPreset(p.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const two = { display: "grid", gridTemplateColumns: "340px 1fr", gap: 12 } as const;

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <TopBar />
      <div style={{ display: "grid", gridTemplateColumns: presenter ? "1fr" : "320px 1fr", gap: 12, padding: 12, flex: 1, overflow: "hidden" }}>
        {!presenter && <ICLab />}
        <div style={{ overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, paddingRight: 4, paddingBottom: presenter ? 80 : 4 }}>
          {training && <LiveTraining />}
          <FieldPanel />
          <ErrorPanel />
          <SnapshotPlot />
          {!presenter && (
            <>
              <div style={two}>
                <Spectrum />
                <MetricsCards />
              </div>
              <div style={two}>
                <ResolutionSweep />
                <CostChart />
              </div>
            </>
          )}
          {presenter && (
            <div style={two}>
              <Spectrum />
              <MetricsCards />
            </div>
          )}
        </div>
      </div>
      <Presenter />
    </div>
  );
}
