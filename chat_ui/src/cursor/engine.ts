type XY = { x: number; y: number };

type CursorPred = {
  type: "pred";
  sample_index?: number;
  gru: XY;
  liquid: XY;
  kalman: XY;
  abg: XY;
  gru_path: XY[];
  liquid_path: XY[];
  kalman_path: XY[];
  abg_path: XY[];
  metrics: {
    gru_mse_ema: number;
    liquid_mse_ema: number;
    rtt_ms: number;
    mse_advantage?: number;
    mode?: string;
    infer_gru_ms_ema?: number | null;
    infer_liquid_ms_ema?: number | null;
    fde250_gru_px_ema?: number;
    fde250_liquid_px_ema?: number;
    fde250_kalman_px_ema?: number;
    fde250_abg_px_ema?: number;
    ade_gru_px_ema?: number;
    ade_liquid_px_ema?: number;
    ade_kalman_px_ema?: number;
    ade_abg_px_ema?: number;
    fde_final_gru_px_ema?: number;
    fde_final_liquid_px_ema?: number;
    fde_final_kalman_px_ema?: number;
    fde_final_abg_px_ema?: number;
    dev_controls_enabled?: boolean;
    debug_gru_bias_px?: number;
  };
  buffer_ready: boolean;
};

export type CursorStats = {
  connected: boolean;
  connectionState: "connecting" | "connected" | "reconnecting" | "disconnected";
  ready: boolean;
  hasServerData: boolean;
  paused: boolean;
  fps: number;
  netRttMs: number;
  inferGruMs: number | null;
  inferLiquidMs: number | null;
  fde250GruPx: number;
  fde250LiquidPx: number;
  fde250KalmanPx: number;
  fde250AbgPx: number;
  adeGruPx: number;
  adeLiquidPx: number;
  adeKalmanPx: number;
  adeAbgPx: number;
  fdeFinalGruPx: number;
  fdeFinalLiquidPx: number;
  fdeFinalKalmanPx: number;
  fdeFinalAbgPx: number;
  liquidTighterPx: number;
  gruMse: number;
  liquidMse: number;
  advantageMse: number;
  rttMs: number;
  mode: string;
  devControlsEnabled: boolean;
  debugGruBiasPx: number;
};

export type CursorEngineHandle = {
  stop: () => void;
  setPaused: (paused: boolean) => void;
  clearTrails: () => void;
  setDebugGruBiasPx: (biasPx: number) => void;
};

function wsProtocolFromPage(): "ws:" | "wss:" {
  return window.location.protocol === "https:" ? "wss:" : "ws:";
}

function normalizePath(path: string): string {
  if (!path) return "/";
  return path.startsWith("/") ? path : `/${path}`;
}

function wsFromRaw(raw: string, defaultPath: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const proto = wsProtocolFromPage();
  const path = normalizePath(defaultPath);

  try {
    if (trimmed.startsWith("/")) {
      return `${proto}//${window.location.host}${trimmed}`;
    }

    const absolute = /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(trimmed)
      ? new URL(trimmed)
      : new URL(`${proto}//${trimmed}`);
    if (absolute.protocol === "http:" || absolute.protocol === "https:") {
      absolute.protocol = absolute.protocol === "https:" ? "wss:" : "ws:";
    } else if (absolute.protocol !== "ws:" && absolute.protocol !== "wss:") {
      return null;
    }
    if (!absolute.pathname || absolute.pathname === "/") {
      absolute.pathname = path;
    }
    return absolute.toString();
  } catch {
    return null;
  }
}

function uniquePush(arr: string[], value: string | null): void {
  if (!value) return;
  if (!arr.includes(value)) arr.push(value);
}

function wsCandidates(path: string): string[] {
  const pathNorm = normalizePath(path);
  const candidates: string[] = [];
  const viteEnv = import.meta.env as Record<string, string | undefined>;

  uniquePush(candidates, wsFromRaw(viteEnv.VITE_CURSOR_WS_URL ?? "", pathNorm));
  uniquePush(candidates, wsFromRaw(viteEnv.VITE_WS_BASE_URL ?? "", pathNorm));
  uniquePush(candidates, wsFromRaw(viteEnv.VITE_API_BASE_URL ?? "", pathNorm));
  uniquePush(candidates, wsFromRaw(`${window.location.host}${pathNorm}`, pathNorm));

  const isViteDevHost =
    (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") &&
    window.location.port === "5173";
  if (isViteDevHost) {
    uniquePush(candidates, wsFromRaw(`127.0.0.1:8000${pathNorm}`, pathNorm));
    uniquePush(candidates, wsFromRaw(`localhost:8000${pathNorm}`, pathNorm));
  }

  return candidates.length > 0 ? candidates : [`${wsProtocolFromPage()}//${window.location.host}${pathNorm}`];
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpPoint(a: XY, b: XY, t: number): XY {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

function pushTail(arr: XY[], p: XY, maxLen = 40): void {
  arr.push({ x: p.x, y: p.y });
  if (arr.length > maxLen) arr.shift();
}

function pathPoint(path: XY[], t01: number): XY {
  if (path.length === 0) return { x: 0, y: 0 };
  if (path.length === 1) return path[0];
  const t = clamp(t01, 0, 1) * (path.length - 1);
  const idx = Math.floor(t);
  const frac = t - idx;
  const p0 = path[Math.max(0, idx - 1)];
  const p1 = path[idx];
  const p2 = path[Math.min(path.length - 1, idx + 1)];
  const p3 = path[Math.min(path.length - 1, idx + 2)];

  const frac2 = frac * frac;
  const frac3 = frac2 * frac;
  const x =
    0.5 *
    ((2 * p1.x) +
      (-p0.x + p2.x) * frac +
      (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * frac2 +
      (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * frac3);
  const y =
    0.5 *
    ((2 * p1.y) +
      (-p0.y + p2.y) * frac +
      (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * frac2 +
      (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * frac3);
  return { x, y };
}

function drawGlowPath(ctx: CanvasRenderingContext2D, path: XY[], color: string, width: number): void {
  if (path.length < 2) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  path.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawTail(ctx: CanvasRenderingContext2D, tail: XY[], color: string): void {
  if (tail.length < 2) return;
  for (let i = 1; i < tail.length; i++) {
    const a = tail[i - 1];
    const b = tail[i];
    const alpha = i / tail.length;
    ctx.strokeStyle = color.replace("{a}", (0.08 + alpha * 0.35).toFixed(3));
    ctx.lineWidth = 0.75 + alpha * 2.2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawRing(ctx: CanvasRenderingContext2D, p: XY, color: string, r: number, pulse: number): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.3;
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r + pulse, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.save();
  ctx.strokeStyle = "rgba(137, 165, 201, 0.075)";
  ctx.lineWidth = 1;
  const grid = 48;
  for (let x = 0; x <= width; x += grid) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += grid) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawLegend(ctx: CanvasRenderingContext2D): void {
  const items = [
    { label: "GRU", color: "rgba(255, 108, 108, 0.95)" },
    { label: "Liquid", color: "rgba(255, 208, 92, 0.95)" },
    { label: "Kalman", color: "rgba(86, 217, 138, 0.95)" },
    { label: "ABG", color: "rgba(255, 159, 69, 0.95)" },
  ];
  const x = 18;
  let y = 20;
  ctx.save();
  ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
  for (const item of items) {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(216, 231, 248, 0.95)";
    ctx.fillText(item.label, x + 10, y + 4);
    y += 18;
  }
  ctx.restore();
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number") return null;
  if (!Number.isFinite(value)) return null;
  return value;
}

export function startCursorEngine(canvas: HTMLCanvasElement, onStats: (stats: CursorStats) => void): CursorEngineHandle {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    const noop = () => {};
    onStats({
      connected: false,
      connectionState: "disconnected",
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
      mode: "unavailable",
      devControlsEnabled: false,
      debugGruBiasPx: 0,
    });
    return { stop: noop, setPaused: noop, clearTrails: noop, setDebugGruBiasPx: noop };
  }

  let mouse: XY = { x: window.innerWidth * 0.5, y: window.innerHeight * 0.5 };
  let lastMouse: XY = { ...mouse };
  let localVel: XY = { x: 0, y: 0 };
  let displayGru: XY = { ...mouse };
  let displayLiquid: XY = { ...mouse };
  let displayKalman: XY = { ...mouse };
  let displayAbg: XY = { ...mouse };
  let gruPath: XY[] = [];
  let liquidPath: XY[] = [];
  let kalmanPath: XY[] = [];
  let abgPath: XY[] = [];
  const gruTail: XY[] = [];
  const liquidTail: XY[] = [];
  const kalmanTail: XY[] = [];
  const abgTail: XY[] = [];

  let ready = false;
  let hasServerData = false;
  let paused = false;
  let connectionState: CursorStats["connectionState"] = "connecting";
  let connected = false;
  let mode = "adaptive-only";

  let netRttMs = 0;
  let netRttInitialized = false;
  let rttMs = 0;
  let inferGruMs: number | null = null;
  let inferLiquidMs: number | null = null;
  let fde250GruPx = 0;
  let fde250LiquidPx = 0;
  let fde250KalmanPx = 0;
  let fde250AbgPx = 0;
  let adeGruPx = 0;
  let adeLiquidPx = 0;
  let adeKalmanPx = 0;
  let adeAbgPx = 0;
  let fdeFinalGruPx = 0;
  let fdeFinalLiquidPx = 0;
  let fdeFinalKalmanPx = 0;
  let fdeFinalAbgPx = 0;
  let gruMse = 0;
  let liquidMse = 0;
  let advantageMse = 0;
  let devControlsEnabled = false;
  let debugGruBiasPx = 0;

  let sampleIndexCounter = 0;
  const sentAtBySample = new Map<number, number>();

  let raf = 0;
  let stopRequested = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: number | null = null;
  let reconnectAttempt = 0;
  const socketUrls = wsCandidates("/ws/cursor");
  let socketUrlIndex = 0;
  let pathReceivedAt = performance.now();
  let lastSendAt = 0;
  let lastFrameAt = performance.now();
  let lastStatsEmit = 0;
  let fps = 0;
  let fpsCounter = 0;
  let fpsWindowStart = performance.now();

  const resize = () => {
    const cssWidth = window.innerWidth;
    const cssHeight = Math.max(320, window.innerHeight - 170);
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  const emitStats = (force = false) => {
    const now = performance.now();
    if (!force && now - lastStatsEmit < 120) return;
    lastStatsEmit = now;
    onStats({
      connected,
      connectionState,
      ready,
      hasServerData,
      paused,
      fps,
      netRttMs,
      inferGruMs,
      inferLiquidMs,
      fde250GruPx,
      fde250LiquidPx,
      fde250KalmanPx,
      fde250AbgPx,
      adeGruPx,
      adeLiquidPx,
      adeKalmanPx,
      adeAbgPx,
      fdeFinalGruPx,
      fdeFinalLiquidPx,
      fdeFinalKalmanPx,
      fdeFinalAbgPx,
      liquidTighterPx: fde250GruPx - fde250LiquidPx,
      gruMse,
      liquidMse,
      advantageMse,
      rttMs,
      mode,
      devControlsEnabled,
      debugGruBiasPx,
    });
  };

  const scheduleReconnect = () => {
    if (stopRequested) return;
    if (reconnectTimer != null) return;
    if (socketUrls.length > 1) {
      socketUrlIndex = (socketUrlIndex + 1) % socketUrls.length;
    }
    connectionState = "reconnecting";
    connected = false;
    emitStats(true);
    const delay = Math.min(2800, 180 * Math.pow(1.8, reconnectAttempt) + Math.random() * 120);
    reconnectAttempt += 1;
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      connectSocket();
    }, delay);
  };

  const connectSocket = () => {
    if (stopRequested) return;
    try {
      connectionState = reconnectAttempt > 0 ? "reconnecting" : "connecting";
      emitStats(true);
      socket = new WebSocket(socketUrls[socketUrlIndex] ?? socketUrls[0]);
    } catch {
      scheduleReconnect();
      return;
    }

    socket.onopen = () => {
      connected = true;
      connectionState = "connected";
      reconnectAttempt = 0;
      emitStats(true);
    };
    socket.onerror = () => {
      connected = false;
      connectionState = "disconnected";
      emitStats(true);
    };
    socket.onclose = () => {
      connected = false;
      connectionState = "disconnected";
      emitStats(true);
      scheduleReconnect();
    };
    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data) as CursorPred;
      if (msg.type !== "pred") return;
      hasServerData = true;

      const now = performance.now();
      const ackSample = asFiniteNumber(msg.sample_index);
      if (ackSample != null) {
        const sentAt = sentAtBySample.get(ackSample);
        if (sentAt != null) {
          const sampleRtt = Math.max(0, now - sentAt);
          if (!netRttInitialized) {
            netRttMs = sampleRtt;
            netRttInitialized = true;
          } else {
            netRttMs = 0.88 * netRttMs + 0.12 * sampleRtt;
          }
        }
        for (const key of sentAtBySample.keys()) {
          if (key <= ackSample) sentAtBySample.delete(key);
        }
      }

      ready = Boolean(msg.buffer_ready);
      gruPath = Array.isArray(msg.gru_path) ? msg.gru_path : gruPath;
      liquidPath = Array.isArray(msg.liquid_path) ? msg.liquid_path : liquidPath;
      kalmanPath = Array.isArray(msg.kalman_path) ? msg.kalman_path : kalmanPath;
      abgPath = Array.isArray(msg.abg_path) ? msg.abg_path : abgPath;
      rttMs = Number(msg.metrics?.rtt_ms ?? rttMs);
      gruMse = Number(msg.metrics?.gru_mse_ema ?? gruMse);
      liquidMse = Number(msg.metrics?.liquid_mse_ema ?? liquidMse);
      advantageMse = Number(msg.metrics?.mse_advantage ?? gruMse - liquidMse);
      mode = String(msg.metrics?.mode ?? mode);

      const inferGru = asFiniteNumber(msg.metrics?.infer_gru_ms_ema);
      const inferLiquid = asFiniteNumber(msg.metrics?.infer_liquid_ms_ema);
      inferGruMs = inferGru;
      inferLiquidMs = inferLiquid;

      fde250GruPx = Number(msg.metrics?.fde250_gru_px_ema ?? fde250GruPx);
      fde250LiquidPx = Number(msg.metrics?.fde250_liquid_px_ema ?? fde250LiquidPx);
      fde250KalmanPx = Number(msg.metrics?.fde250_kalman_px_ema ?? fde250KalmanPx);
      fde250AbgPx = Number(msg.metrics?.fde250_abg_px_ema ?? fde250AbgPx);
      adeGruPx = Number(msg.metrics?.ade_gru_px_ema ?? adeGruPx);
      adeLiquidPx = Number(msg.metrics?.ade_liquid_px_ema ?? adeLiquidPx);
      adeKalmanPx = Number(msg.metrics?.ade_kalman_px_ema ?? adeKalmanPx);
      adeAbgPx = Number(msg.metrics?.ade_abg_px_ema ?? adeAbgPx);
      fdeFinalGruPx = Number(msg.metrics?.fde_final_gru_px_ema ?? fdeFinalGruPx);
      fdeFinalLiquidPx = Number(msg.metrics?.fde_final_liquid_px_ema ?? fdeFinalLiquidPx);
      fdeFinalKalmanPx = Number(msg.metrics?.fde_final_kalman_px_ema ?? fdeFinalKalmanPx);
      fdeFinalAbgPx = Number(msg.metrics?.fde_final_abg_px_ema ?? fdeFinalAbgPx);
      devControlsEnabled = Boolean(msg.metrics?.dev_controls_enabled ?? devControlsEnabled);

      pathReceivedAt = now;
      emitStats();
    };
  };

  const onMouseMove = (ev: MouseEvent) => {
    mouse = { x: ev.clientX, y: ev.clientY };
  };

  const clearTrails = () => {
    gruTail.length = 0;
    liquidTail.length = 0;
    kalmanTail.length = 0;
    abgTail.length = 0;
  };

  resize();
  window.addEventListener("resize", resize);
  window.addEventListener("mousemove", onMouseMove);
  connectSocket();

  const frame = () => {
    const now = performance.now();
    const dt = Math.max(1 / 240, Math.min((now - lastFrameAt) / 1000, 0.08));
    lastFrameAt = now;

    fpsCounter += 1;
    if (now - fpsWindowStart >= 1000) {
      fps = Math.round((fpsCounter * 1000) / (now - fpsWindowStart));
      fpsCounter = 0;
      fpsWindowStart = now;
    }

    const mouseV = { x: (mouse.x - lastMouse.x) / Math.max(dt, 1e-3), y: (mouse.y - lastMouse.y) / Math.max(dt, 1e-3) };
    lastMouse = { ...mouse };
    localVel = {
      x: lerp(localVel.x, mouseV.x, 0.18),
      y: lerp(localVel.y, mouseV.y, 0.18),
    };

    if (!paused && connected && socket && socket.readyState === WebSocket.OPEN && now - lastSendAt > 16) {
      lastSendAt = now;
      const sampleIndex = sampleIndexCounter;
      sampleIndexCounter += 1;
      sentAtBySample.set(sampleIndex, now);
      if (sentAtBySample.size > 512) {
        const oldest = sentAtBySample.keys().next().value;
        if (typeof oldest === "number") sentAtBySample.delete(oldest);
      }

      const payload: Record<string, number | string> = {
        type: "frame",
        sample_index: sampleIndex,
        x: mouse.x,
        y: mouse.y,
        t_ms: Date.now(),
        viewport_w: window.innerWidth,
        viewport_h: window.innerHeight,
      };
      if (debugGruBiasPx !== 0) payload.debug_gru_bias_px = debugGruBiasPx;
      socket.send(JSON.stringify(payload));
    }

    const stale = now - pathReceivedAt > 380;
    const hasServerGru = !stale && gruPath.length > 0;
    const hasServerLiquid = !stale && liquidPath.length > 0;
    const hasServerKalman = !stale && kalmanPath.length > 0;
    const hasServerAbg = !stale && abgPath.length > 0;
    const progress = clamp((now - pathReceivedAt) / 170, 0, 1);
    const localFallbackGru = hasServerData
      ? { x: mouse.x + localVel.x * 0.18, y: mouse.y + localVel.y * 0.18 }
      : mouse;
    const localFallbackLiquid = hasServerData
      ? { x: mouse.x + localVel.x * 0.12, y: mouse.y + localVel.y * 0.12 }
      : mouse;
    const localFallbackKalman = hasServerData
      ? { x: mouse.x + localVel.x * 0.16, y: mouse.y + localVel.y * 0.16 }
      : mouse;
    const localFallbackAbg = hasServerData
      ? { x: mouse.x + localVel.x * 0.14, y: mouse.y + localVel.y * 0.14 }
      : mouse;
    const targetGru = hasServerGru
      ? pathPoint(gruPath, progress)
      : localFallbackGru;
    const targetLiquid = hasServerLiquid
      ? pathPoint(liquidPath, progress)
      : localFallbackLiquid;
    const targetKalman = hasServerKalman
      ? pathPoint(kalmanPath, progress)
      : localFallbackKalman;
    const targetAbg = hasServerAbg
      ? pathPoint(abgPath, progress)
      : localFallbackAbg;

    const smooth = paused ? 0.06 : 0.23;
    displayGru = lerpPoint(displayGru, targetGru, smooth);
    displayLiquid = lerpPoint(displayLiquid, targetLiquid, smooth * 1.08);
    displayKalman = lerpPoint(displayKalman, targetKalman, smooth * 1.02);
    displayAbg = lerpPoint(displayAbg, targetAbg, smooth * 1.02);
    pushTail(gruTail, displayGru, 34);
    pushTail(liquidTail, displayLiquid, 34);
    pushTail(kalmanTail, displayKalman, 30);
    pushTail(abgTail, displayAbg, 30);

    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const bg = ctx.createRadialGradient(cssWidth * 0.5, cssHeight * 0.1, 30, cssWidth * 0.5, cssHeight * 0.6, cssHeight);
    bg.addColorStop(0, "#111b2d");
    bg.addColorStop(0.5, "#0c131d");
    bg.addColorStop(1, "#080b11");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    drawGrid(ctx, cssWidth, cssHeight);
    drawLegend(ctx);

    drawTail(ctx, gruTail, "rgba(255,100,100,{a})");
    drawTail(ctx, liquidTail, "rgba(84,201,255,{a})");
    drawTail(ctx, kalmanTail, "rgba(85,215,155,{a})");
    drawTail(ctx, abgTail, "rgba(255,168,86,{a})");
    drawGlowPath(ctx, gruPath, "rgba(255, 108, 108, 0.86)", 1.45);
    drawGlowPath(ctx, liquidPath, "rgba(84, 201, 255, 0.9)", 1.45);
    drawGlowPath(ctx, kalmanPath, "rgba(86, 217, 138, 0.78)", 1.2);
    drawGlowPath(ctx, abgPath, "rgba(255, 159, 69, 0.78)", 1.2);

    const pulse = Math.sin(now / 170) * 0.9;
    drawRing(ctx, displayGru, "rgba(255, 108, 108, 0.92)", 13, pulse);
    drawRing(ctx, displayLiquid, "rgba(255, 208, 92, 0.95)", 13, -pulse);
    drawRing(ctx, displayKalman, "rgba(86, 217, 138, 0.9)", 10.5, pulse * 0.6);
    drawRing(ctx, displayAbg, "rgba(255, 159, 69, 0.9)", 10.5, -pulse * 0.6);

    ctx.save();
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 12;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(mouse.x, mouse.y, 3.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    emitStats();
    raf = window.requestAnimationFrame(frame);
  };
  raf = window.requestAnimationFrame(frame);

  const stop = () => {
    stopRequested = true;
    if (reconnectTimer != null) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    window.cancelAnimationFrame(raf);
    window.removeEventListener("resize", resize);
    window.removeEventListener("mousemove", onMouseMove);
    sentAtBySample.clear();
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      socket.close();
    }
    connected = false;
    connectionState = "disconnected";
    emitStats(true);
  };

  return {
    stop,
    setPaused: (v: boolean) => {
      paused = Boolean(v);
      emitStats(true);
    },
    clearTrails,
    setDebugGruBiasPx: (biasPx: number) => {
      debugGruBiasPx = Number.isFinite(biasPx) ? biasPx : 0;
      emitStats(true);
    },
  };
}
