import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
LOGS = {
    "FakeFilmBW": ROOT / "logs" / "fakefilmbw.log",
    "FakeFilmColor": ROOT / "logs" / "fakefilmcolor.log",
}
CHECKPOINTS = {
    "FakeFilmBW": [
        ROOT / "checkpoints" / "fakefilmbw_deeplab_best.pth",
        ROOT / "checkpoints" / "fakefilmbw_deeplab_latest.pth",
    ],
    "FakeFilmColor": [
        ROOT / "checkpoints" / "fakefilmcolor_deeplab_best.pth",
        ROOT / "checkpoints" / "fakefilmcolor_deeplab_latest.pth",
    ],
}

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
TRAIN_METRIC_RE = re.compile(r"train loss=([0-9.]+) iou=([0-9.]+) dice=([0-9.]+)")
VAL_METRIC_RE = re.compile(r"val\s+loss=([0-9.]+) iou=([0-9.]+) dice=([0-9.]+)")
PROGRESS_RE = re.compile(r"(Training|Validation):\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)")
BEST_RE = re.compile(r"Best Dice:\s+([0-9.]+)")


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FakeFilm Training Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1b1f23;
      --muted: #65717c;
      --line: #d7dde3;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --green: #16784b;
      --red: #b42318;
      --blue: #1f6feb;
      --pink: #c43a5b;
      --yellow: #b7791f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0 0 6px; font-size: 24px; }
    .sub { color: var(--muted); font-size: 14px; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      padding: 18px 28px 32px;
      max-width: 1280px;
      margin: 0 auto;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }
    h2 { margin: 0; font-size: 18px; }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .live { color: var(--green); border-color: #7bc6a3; }
    .waiting { color: var(--yellow); border-color: #e1c16e; }
    .done { color: var(--blue); border-color: #8ab4f8; }
    .failed { color: var(--red); border-color: #f1a5a0; }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 70px;
    }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 22px; font-weight: 650; margin-top: 6px; }
    .bar {
      height: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #eef1f4;
      margin: 8px 0 12px;
    }
    .fill { height: 100%; background: var(--green); width: 0%; transition: width 0.25s ease; }
    canvas {
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 230px;
      overflow: auto;
      margin: 10px 0 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 880px) {
      header { padding: 18px 16px 12px; }
      main { padding: 14px 16px 24px; }
      .grid, .stats { grid-template-columns: 1fr; }
      .row { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
<header>
  <h1>FakeFilm Training</h1>
  <div class="sub">自动刷新：<span id="updated">-</span></div>
</header>
<main>
  <section class="grid" id="cards"></section>
  <section class="card">
    <div class="row">
      <h2>Validation Dice</h2>
      <span class="badge">两个模型对比</span>
    </div>
    <canvas id="chartDice"></canvas>
  </section>
  <section class="card">
    <div class="row">
      <h2>Validation Loss</h2>
      <span class="badge">越低越好</span>
    </div>
    <canvas id="chartLoss"></canvas>
  </section>
</main>
<script>
const colors = { FakeFilmBW: "#1f6feb", FakeFilmColor: "#c43a5b" };

function statusClass(status) {
  if (status === "running") return "live";
  if (status === "finished") return "done";
  if (status === "failed") return "failed";
  return "waiting";
}

function statusText(status) {
  return {
    running: "训练中",
    finished: "已完成",
    failed: "失败",
    waiting: "等待中"
  }[status] || status;
}

function fmt(v, digits = 4) {
  return Number.isFinite(v) ? v.toFixed(digits) : "-";
}

function card(model) {
  const pct = model.progress?.percent ?? 0;
  const last = model.lastMetrics || {};
  const ckpt = model.checkpoints || [];
  const ckptText = ckpt.length ? ckpt.map(c => `${c.name}: ${c.time || "-"} ${(c.mb || 0).toFixed(1)} MB`).join("\n") : "暂无 checkpoint";
  return `
    <article class="card">
      <div class="row">
        <h2>${model.name}</h2>
        <span class="badge ${statusClass(model.status)}">${statusText(model.status)}</span>
      </div>
      <div class="label">Epoch ${model.epoch || 0}/${model.totalEpochs || 0} · ${model.progress?.phase || "-"}</div>
      <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
      <div class="label">${model.progress?.done || 0}/${model.progress?.total || 0} batches · ${pct}%</div>
      <div class="stats">
        <div class="stat"><div class="label">Train Loss</div><div class="value">${fmt(last.trainLoss)}</div></div>
        <div class="stat"><div class="label">Val Loss</div><div class="value">${fmt(last.valLoss)}</div></div>
        <div class="stat"><div class="label">Val IoU</div><div class="value">${fmt(last.valIou)}</div></div>
        <div class="stat"><div class="label">Val Dice</div><div class="value">${fmt(last.valDice)}</div></div>
      </div>
      <div class="label">Best Dice: ${fmt(model.bestDice)}</div>
      <pre>${ckptText}</pre>
      <pre>${model.tail.join("\n")}</pre>
    </article>
  `;
}

function drawChart(canvas, models, field, title) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 46, r: 16, t: 18, b: 34 };
  const series = models.map(m => ({
    name: m.name,
    points: (m.metrics || []).map(x => ({ x: x.epoch, y: x[field] })).filter(p => Number.isFinite(p.y))
  }));
  const all = series.flatMap(s => s.points);
  ctx.strokeStyle = "#d7dde3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
  ctx.fillStyle = "#65717c";
  ctx.font = "12px Segoe UI, Arial";
  if (!all.length) {
    ctx.fillText("等待第一个 epoch 完成", pad.l + 12, pad.t + 26);
    return;
  }
  const maxX = Math.max(...all.map(p => p.x), 1);
  let minY = Math.min(...all.map(p => p.y));
  let maxY = Math.max(...all.map(p => p.y));
  if (minY === maxY) { minY = Math.max(0, minY - 0.1); maxY += 0.1; }
  const xAt = x => pad.l + (x - 1) / Math.max(1, maxX - 1) * (w - pad.l - pad.r);
  const yAt = y => h - pad.b - (y - minY) / (maxY - minY) * (h - pad.t - pad.b);
  ctx.fillText(maxY.toFixed(3), 8, pad.t + 4);
  ctx.fillText(minY.toFixed(3), 8, h - pad.b);
  ctx.fillText("epoch", w - 54, h - 10);
  series.forEach((s, idx) => {
    if (!s.points.length) return;
    ctx.strokeStyle = colors[s.name] || ["#1f6feb", "#c43a5b"][idx % 2];
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = xAt(p.x), y = yAt(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    s.points.forEach(p => {
      ctx.beginPath();
      ctx.arc(xAt(p.x), yAt(p.y), 3, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.fillText(s.name, pad.l + idx * 130, 14);
  });
}

async function refresh() {
  const res = await fetch("/api/status", { cache: "no-store" });
  const data = await res.json();
  const models = Object.values(data.models);
  document.getElementById("updated").textContent = new Date(data.now * 1000).toLocaleString();
  document.getElementById("cards").innerHTML = models.map(card).join("");
  drawChart(document.getElementById("chartDice"), models, "valDice");
  drawChart(document.getElementById("chartLoss"), models, "valLoss");
}

refresh();
setInterval(refresh, 5000);
window.addEventListener("resize", refresh);
</script>
</body>
</html>
"""


def tail_lines(path, limit=80):
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= limit:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-limit:]


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def checkpoint_info(paths):
    items = []
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        items.append({
            "name": path.name,
            "mb": stat.st_size / 1024 / 1024,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        })
    return items


def parse_log(name, path):
    text = read_text(path)
    lines = text.splitlines()
    epoch = 0
    total_epochs = 0
    metrics = []
    pending_train = None
    progress = {"phase": "-", "percent": 0, "done": 0, "total": 0}
    best_dice = None

    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))

        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            progress = {
                "phase": progress_match.group(1),
                "percent": int(progress_match.group(2)),
                "done": int(progress_match.group(3)),
                "total": int(progress_match.group(4)),
            }

        train_match = TRAIN_METRIC_RE.search(line)
        if train_match:
            pending_train = {
                "epoch": epoch,
                "trainLoss": float(train_match.group(1)),
                "trainIou": float(train_match.group(2)),
                "trainDice": float(train_match.group(3)),
            }

        val_match = VAL_METRIC_RE.search(line)
        if val_match:
            row = pending_train.copy() if pending_train else {"epoch": epoch}
            row.update({
                "valLoss": float(val_match.group(1)),
                "valIou": float(val_match.group(2)),
                "valDice": float(val_match.group(3)),
            })
            metrics.append(row)
            pending_train = None

        best_match = BEST_RE.search(line)
        if best_match:
            best_dice = float(best_match.group(1))

    if metrics and best_dice is None:
        best_dice = max(row.get("valDice", 0.0) for row in metrics)

    status = "waiting"
    if path.exists():
        status = "running"
    if "Done. Best Dice:" in text:
        status = "finished"
        progress = {"phase": "Done", "percent": 100, "done": progress.get("total", 0), "total": progress.get("total", 0)}
    if "failed with exit code" in text or "Traceback (most recent call last)" in text:
        status = "failed"

    return {
        "name": name,
        "status": status,
        "epoch": epoch,
        "totalEpochs": total_epochs,
        "progress": progress,
        "metrics": metrics,
        "lastMetrics": metrics[-1] if metrics else {},
        "bestDice": best_dice,
        "tail": tail_lines(path, 18),
        "checkpoints": checkpoint_info(CHECKPOINTS[name]),
    }


def status_payload():
    return {
        "now": time.time(),
        "models": {name: parse_log(name, path) for name, path in LOGS.items()},
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            body = json.dumps(status_payload(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        return


def main():
    port = int(os.environ.get("TRAINING_DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Training dashboard: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
