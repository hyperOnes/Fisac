import { useEffect, useRef, useState } from "react";

import { CursorEngineHandle, CursorStats, startCursorEngine } from "../cursor/engine";

const INIT_STATS: CursorStats = {
  connected: false,
  connectionState: "connecting",
  ready: false,
  hasServerData: false,
  paused: false,
  fps: 0,
  netRttMs: 0,
  inferGruMs: null,
  inferLiquidMs: null,
  fde250GruPx: 0,
  fde250LiquidPx: 0,
  fde250KalmanPx: 0,
  fde250AbgPx: 0,
  adeGruPx: 0,
  adeLiquidPx: 0,
  adeKalmanPx: 0,
  adeAbgPx: 0,
  fdeFinalGruPx: 0,
  fdeFinalLiquidPx: 0,
  fdeFinalKalmanPx: 0,
  fdeFinalAbgPx: 0,
  liquidTighterPx: 0,
  gruMse: 0,
  liquidMse: 0,
  advantageMse: 0,
  rttMs: 0,
  mode: "adaptive-only",
  devControlsEnabled: false,
  debugGruBiasPx: 0,
};

function formatMs(value: number | null): string {
  if (value == null) return "N/A";
  return `${value.toFixed(2)} ms`;
}

function formatMetricPx(value: number, enabled: boolean): string {
  if (!enabled) return "N/A";
  return `${value.toFixed(2)} px`;
}

export function CursorLabPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<CursorEngineHandle | null>(null);
  const [stats, setStats] = useState<CursorStats>(INIT_STATS);
  const [paused, setPaused] = useState(false);
  const [gruBiasEnabled, setGruBiasEnabled] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const engine = startCursorEngine(canvas, (next) => setStats(next));
    engineRef.current = engine;
    return () => {
      engine.stop();
      engineRef.current = null;
    };
  }, []);

  const perfTone = stats.fps >= 55 ? "pill-on" : "pill-off";
  const tighterTone = stats.liquidTighterPx >= 0 ? "pill-on" : "pill-off";
  const metricsLive = stats.connected && stats.hasServerData;
  const liquidVsKalman = metricsLive ? stats.fde250KalmanPx - stats.fde250LiquidPx : 0;
  const liquidVsAbg = metricsLive ? stats.fde250AbgPx - stats.fde250LiquidPx : 0;
  const bufferState = !stats.connected ? "offline" : stats.ready ? "ready" : "warming";

  return (
    <section className="cursor-lab-wrap">
      <div className="cursor-lab-header">
        <div className="cursor-lab-title">
          <h2>Cursor Lab</h2>
          <p>Realtime 5-point trajectory prediction with px-native scoring and deterministic sample alignment.</p>
        </div>
        <div className="cursor-lab-controls">
          <button
            className={`btn ${paused ? "" : "btn-primary"}`}
            onClick={() => {
              const next = !paused;
              setPaused(next);
              engineRef.current?.setPaused(next);
            }}
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            className="btn"
            onClick={() => {
              engineRef.current?.clearTrails();
            }}
          >
            Clear Trails
          </button>
          <button
            className={`btn ${gruBiasEnabled ? "btn-primary" : ""}`}
            disabled={!stats.connected}
            onClick={() => {
              const next = !gruBiasEnabled;
              setGruBiasEnabled(next);
              engineRef.current?.setDebugGruBiasPx(next ? 20 : 0);
            }}
          >
            {gruBiasEnabled ? "Disable +20px GRU Bias" : "Inject +20px GRU Bias"}
          </button>
        </div>
      </div>

      <div className="cursor-lab-stats">
        <span className={`pill ${stats.connected ? "pill-on" : "pill-off"}`}>WS {stats.connectionState}</span>
        <span className={`pill ${stats.ready && stats.connected ? "pill-on" : "pill-off"}`}>buffer {bufferState}</span>
        <span className={`pill ${perfTone}`}>FPS {stats.fps}</span>
        <span className="pill">Net RTT {metricsLive ? `${stats.netRttMs.toFixed(2)} ms` : "N/A"}</span>
        <span className="pill">
          Infer G/L {metricsLive ? formatMs(stats.inferGruMs) : "N/A"} / {metricsLive ? formatMs(stats.inferLiquidMs) : "N/A"}
        </span>
        <span className="pill">FDE@250 G/L {formatMetricPx(stats.fde250GruPx, metricsLive)} / {formatMetricPx(stats.fde250LiquidPx, metricsLive)}</span>
        <span className="pill">ADE G/L {formatMetricPx(stats.adeGruPx, metricsLive)} / {formatMetricPx(stats.adeLiquidPx, metricsLive)}</span>
        <span className={`pill ${tighterTone}`}>Liquid tighter by {metricsLive ? `${stats.liquidTighterPx.toFixed(2)} px` : "N/A"}</span>
        <span className="pill">Kalman FDE@250 {formatMetricPx(stats.fde250KalmanPx, metricsLive)}</span>
        <span className="pill">ABG FDE@250 {formatMetricPx(stats.fde250AbgPx, metricsLive)}</span>
        <span className={`pill ${liquidVsKalman >= 0 ? "pill-on" : "pill-off"}`}>Liquid vs Kalman {metricsLive ? `${liquidVsKalman.toFixed(2)} px` : "N/A"}</span>
        <span className={`pill ${liquidVsAbg >= 0 ? "pill-on" : "pill-off"}`}>Liquid vs ABG {metricsLive ? `${liquidVsAbg.toFixed(2)} px` : "N/A"}</span>
        <span className="pill">mode {stats.mode}</span>
      </div>

      {import.meta.env.DEV && (
        <div className="cursor-lab-stats">
          <span className="pill">GRU MSE {metricsLive ? stats.gruMse.toExponential(2) : "N/A"}</span>
          <span className="pill">Liquid MSE {metricsLive ? stats.liquidMse.toExponential(2) : "N/A"}</span>
          <span className={`pill ${stats.advantageMse >= 0 ? "pill-on" : "pill-off"}`}>ΔMSE {metricsLive ? stats.advantageMse.toExponential(2) : "N/A"}</span>
          <span className="pill">FDE Final G/L {formatMetricPx(stats.fdeFinalGruPx, metricsLive)} / {formatMetricPx(stats.fdeFinalLiquidPx, metricsLive)}</span>
          <span className="pill">ADE Kalman/ABG {formatMetricPx(stats.adeKalmanPx, metricsLive)} / {formatMetricPx(stats.adeAbgPx, metricsLive)}</span>
          <span className="pill">FDE Final Kalman/ABG {formatMetricPx(stats.fdeFinalKalmanPx, metricsLive)} / {formatMetricPx(stats.fdeFinalAbgPx, metricsLive)}</span>
          <span className="pill">Compat RTT {metricsLive ? `${stats.rttMs.toFixed(2)} ms` : "N/A"}</span>
          <span className={`pill ${stats.devControlsEnabled ? "pill-on" : "pill-off"}`}>dev controls {stats.devControlsEnabled ? "enabled" : "off"}</span>
        </div>
      )}

      <canvas ref={canvasRef} className="cursor-stage" />
    </section>
  );
}
