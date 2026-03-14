(function () {
  const canvas = document.getElementById("stage");
  const stats = document.getElementById("stats");
  const ctx = canvas.getContext("2d");

  let mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
  let gru = { x: mouse.x, y: mouse.y };
  let liquid = { x: mouse.x, y: mouse.y };
  let gruPath = [];
  let liquidPath = [];
  let metrics = { gru_mse_ema: 0, liquid_mse_ema: 0, rtt_ms: 0 };
  const gruTail = [];
  const liquidTail = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/cursor`);

  window.addEventListener("mousemove", (ev) => {
    mouse = { x: ev.clientX, y: ev.clientY };
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "frame",
          x: mouse.x,
          y: mouse.y,
          t_ms: Date.now(),
          viewport_w: window.innerWidth,
          viewport_h: window.innerHeight,
        })
      );
    }
  });

  ws.onopen = () => {
    stats.textContent = "Cursor WS connected";
  };
  ws.onerror = () => {
    stats.textContent = "Cursor WS error";
  };
  ws.onclose = () => {
    stats.textContent = "Cursor WS closed";
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== "pred") {
      return;
    }
    gru = msg.gru || gru;
    liquid = msg.liquid || liquid;
    gruPath = msg.gru_path || [];
    liquidPath = msg.liquid_path || [];
    metrics = msg.metrics || metrics;
    pushTail(gruTail, gru);
    pushTail(liquidTail, liquid);
    const ready = Boolean(msg.buffer_ready);
    stats.textContent = `buffer_ready=${ready} | gru_mse=${metrics.gru_mse_ema.toFixed(4)} | liquid_mse=${metrics.liquid_mse_ema.toFixed(4)} | rtt=${metrics.rtt_ms.toFixed(1)}ms`;
  };

  function pushTail(arr, point) {
    arr.push({ x: point.x, y: point.y });
    if (arr.length > 10) {
      arr.shift();
    }
  }

  function drawTail(arr, strokeStyle) {
    if (arr.length < 2) return;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const p = arr[i];
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function drawPath(path, strokeStyle) {
    if (!Array.isArray(path) || path.length === 0) return;
    ctx.beginPath();
    path.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawRing(point, color, radius) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function drawDot(point, color, radius) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function frame() {
    ctx.fillStyle = "#0b0d10";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    drawTail(gruTail, "rgba(255, 90, 90, 0.45)");
    drawTail(liquidTail, "rgba(255, 209, 102, 0.45)");
    drawPath(gruPath, "rgba(255, 80, 80, 0.65)");
    drawPath(liquidPath, "rgba(80, 190, 255, 0.65)");

    drawRing(gru, "rgba(255, 90, 90, 0.85)", 14);
    drawRing(liquid, "rgba(80, 190, 255, 0.9)", 14);
    drawDot(mouse, "#ffffff", 4);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
