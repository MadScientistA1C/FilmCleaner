import argparse
import base64
import cgi
import io
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


MASK_ENHANCE_CHOICES = {"none", "sharpen"}
TTA_TYPE_CHOICES = {"flip", "contrast", "gamma", "brightness", "rotate", "scale", "multiscale"}
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
SUFFIX_BY_FORMAT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "MPO": ".jpg",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LAMALocal 闄ゅ皹</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #18211d; background: #f3f5f2; }
body { display: grid; grid-template-rows: auto 1fr; }
.bar { min-height: 58px; display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #fff; border-bottom: 1px solid #d3ddd6; flex-wrap: wrap; }
.brand { font-weight: 700; margin-right: 8px; }
button, select, input[type="file"]::file-selector-button { border: 1px solid #89988f; background: #fff; color: #18211d; border-radius: 6px; padding: 7px 11px; cursor: pointer; }
button.primary { background: #12624f; border-color: #12624f; color: #fff; }
button:disabled, select:disabled, input:disabled { opacity: .55; cursor: wait; }
label { display: inline-flex; align-items: center; gap: 7px; }
input[type="range"] { width: 130px; }
#status { color: #516059; min-width: 220px; }
.stage { min-height: 0; overflow: auto; display: grid; place-items: start center; padding: 18px; }
.canvasWrap { position: relative; line-height: 0; background: #202522; box-shadow: 0 1px 9px rgba(0,0,0,.18); }
canvas { position: absolute; left: 0; top: 0; }
#imageCanvas { position: static; }
#maskCanvas { opacity: .48; cursor: crosshair; touch-action: none; }
.brushCursor { position: absolute; left: 0; top: 0; width: 0; height: 0; border: 1px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,.82); pointer-events: none; display: none; z-index: 4; transform: translate(-50%, -50%); }
.brushCursor.erase { border-style: dashed; }
.empty { margin-top: 14vh; display: grid; gap: 14px; justify-items: center; color: #55635c; }
.empty strong { font-size: 20px; color: #18211d; }
.busy { position: fixed; inset: 0; background: rgba(243,245,242,.78); display: none; place-items: center; z-index: 20; }
.busy.active { display: grid; }
.panel { background: #fff; border: 1px solid #c9d4ce; border-radius: 8px; padding: 20px 24px; display: grid; gap: 12px; justify-items: center; box-shadow: 0 10px 34px rgba(0,0,0,.16); }
.spinner { width: 38px; height: 38px; border: 4px solid #cbd7d0; border-top-color: #12624f; border-radius: 50%; animation: spin .8s linear infinite; }
.progressTrack { width: min(420px, 72vw); height: 10px; border: 1px solid #b7c4bd; border-radius: 999px; overflow: hidden; background: #edf1ee; }
.progressFill { height: 100%; width: 0%; background: #12624f; transition: width .18s ease; }
.progressText { font-size: 13px; color: #516059; min-height: 18px; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="bar">
  <span class="brand">LAMALocal</span>
  <input id="file" type="file" accept="image/*">
  <label>绫诲瀷 <select id="kind"><option value="color">褰╄壊</option><option value="bw">榛戠櫧</option></select></label>
  <button id="auto" class="primary">涓€閿櫎灏?/button>
  <button id="manual" class="primary">鎵嬪姩闄ゅ皹</button>
  <button id="compare">瀵规瘮</button>
  <button id="undo">鎾ら攢</button>
  <button id="clear">娓呯┖鏍囪</button>
  <button id="zoomOut">缂╁皬</button>
  <button id="zoomIn">鏀惧ぇ</button>
  <button id="fit">閫傞厤</button>
  <label>鐢荤瑪 <input id="brush" type="range" min="2" max="160" value="34"><span id="brushValue">34</span>px</label>
  <button id="download">涓嬭浇缁撴灉</button>
  <span id="zoomValue">100%</span>
  <span id="status">璇峰厛涓婁紶鍥剧墖</span>
</div>
<div class="stage">
  <div id="empty" class="empty"><strong>涓婁紶鍥剧墖寮€濮嬮櫎灏?/strong><span>鍙竴閿嚜鍔ㄥ鐞嗭紝涔熷彲鐢ㄧ孩鑹茬敾绗旀秱鎶瑰悗鎵嬪姩淇銆?/span></div>
  <div id="wrap" class="canvasWrap" hidden>
    <canvas id="imageCanvas"></canvas>
    <canvas id="maskCanvas"></canvas>
  </div>
</div>
<div id="busy" class="busy"><div class="panel"><div class="spinner"></div><div id="busyText">姝ｅ湪澶勭悊...</div></div></div>
<script>
const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const imageCtx = imageCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const wrap = document.getElementById('wrap');
const empty = document.getElementById('empty');
const statusEl = document.getElementById('status');
const busy = document.getElementById('busy');
const busyText = document.getElementById('busyText');
const brush = document.getElementById('brush');
const controls = [...document.querySelectorAll('button, select, input')];
let drawing = false;
let history = [];
let zoom = 1;
let hasImage = false;
let hasResult = false;
let compareOriginal = false;
let previewScale = 1;

function setBusy(on, text) {
  busy.classList.toggle('active', on);
  busyText.textContent = text || '姝ｅ湪澶勭悊...';
  controls.forEach(el => el.disabled = on);
}
function applyZoom() {
  wrap.style.transform = `scale(${zoom})`;
  wrap.style.transformOrigin = 'top left';
  wrap.style.marginRight = `${imageCanvas.width * (zoom - 1)}px`;
  wrap.style.marginBottom = `${imageCanvas.height * (zoom - 1)}px`;
  document.getElementById('zoomValue').textContent = `${Math.round(zoom * 100)}%`;
}
function setZoom(next) {
  zoom = Math.max(0.1, Math.min(6, next));
  applyZoom();
}
function fitToStage() {
  const stage = document.querySelector('.stage');
  const zx = (stage.clientWidth - 36) / Math.max(imageCanvas.width, 1);
  const zy = (stage.clientHeight - 36) / Math.max(imageCanvas.height, 1);
  setZoom(Math.min(1, zx, zy));
}
function pushHistory() {
  if (!hasImage) return;
  history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  if (history.length > 30) history.shift();
}
function point(evt) {
  const r = maskCanvas.getBoundingClientRect();
  return { x: (evt.clientX - r.left) * maskCanvas.width / r.width, y: (evt.clientY - r.top) * maskCanvas.height / r.height };
}
function paint(evt) {
  if (!drawing || !hasImage) return;
  const p = point(evt);
  maskCtx.fillStyle = 'rgba(255,0,0,1)';
  maskCtx.beginPath();
  maskCtx.arc(p.x, p.y, Number(brush.value), 0, Math.PI * 2);
  maskCtx.fill();
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      imageCanvas.width = maskCanvas.width = img.naturalWidth;
      imageCanvas.height = maskCanvas.height = img.naturalHeight;
      imageCtx.drawImage(img, 0, 0);
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      wrap.hidden = false;
      empty.hidden = true;
      hasImage = true;
      fitToStage();
      resolve();
    };
    img.onerror = reject;
    img.src = src;
  });
}
async function postJson(url, payload, text) {
  setBusy(true, text);
  try {
    const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload || {}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '澶勭悊澶辫触');
    if (data.preview) {
      hasResult = data.hasResult !== false;
      compareOriginal = false;
      await loadImage(data.preview + '?t=' + Date.now());
    }
    statusEl.textContent = data.message || '澶勭悊瀹屾垚';
    return data;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    setBusy(false);
  }
}
async function postForm(url, formData, text) {
  setBusy(true, text);
  try {
    const response = await fetch(url, { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload failed');
    if (data.preview) {
      hasResult = data.hasResult !== false;
      compareOriginal = false;
      previewScale = data.previewScale || 1;
      await loadImage(data.preview + '?t=' + Date.now());
    }
    statusEl.textContent = data.message || 'Done';
    return data;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    setBusy(false);
  }
}
document.getElementById('file').onchange = async evt => {
  const file = evt.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('image', file, file.name);
  await postForm('/upload', formData, 'Loading image...');
};
brush.oninput = () => document.getElementById('brushValue').textContent = brush.value;
maskCanvas.addEventListener('pointerdown', evt => { drawing = true; pushHistory(); maskCanvas.setPointerCapture(evt.pointerId); paint(evt); });
maskCanvas.addEventListener('pointermove', paint);
maskCanvas.addEventListener('pointerup', () => drawing = false);
maskCanvas.addEventListener('pointercancel', () => drawing = false);
document.getElementById('auto').onclick = () => postJson('/auto', { kind: document.getElementById('kind').value }, '姝ｅ湪涓€閿櫎灏?..');
document.getElementById('manual').onclick = () => postJson('/manual', { mask: maskCanvas.toDataURL('image/png') }, '姝ｅ湪鎵嬪姩闄ゅ皹...');
document.getElementById('undo').onclick = () => {
  const last = history.pop();
  if (last) { maskCtx.putImageData(last, 0, 0); return; }
  postJson('/undo-result', {}, '姝ｅ湪鎾ら攢缁撴灉...');
};
document.getElementById('clear').onclick = () => { pushHistory(); maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); };
document.getElementById('zoomIn').onclick = () => setZoom(zoom * 1.25);
document.getElementById('zoomOut').onclick = () => setZoom(zoom / 1.25);
document.getElementById('fit').onclick = () => {
  fitToStage();
};
document.getElementById('compare').onclick = async () => {
  if (!hasResult) return;
  compareOriginal = !compareOriginal;
  await loadImage((compareOriginal ? '/image' : '/result') + '?t=' + Date.now());
  statusEl.textContent = compareOriginal ? '姝ｅ湪鏌ョ湅鍘熷浘' : '姝ｅ湪鏌ョ湅缁撴灉';
};
document.getElementById('download').onclick = () => {
  if (hasResult) window.location.href = '/download?t=' + Date.now();
};
document.getElementById('langToggle').onclick = () => {
  language = language === 'zh' ? 'en' : 'zh';
  localStorage.setItem('lamaLanguage', language);
  applyLanguage();
};
document.querySelectorAll('.shortcutInput').forEach(input => {
  input.addEventListener('keydown', evt => {
    evt.preventDefault();
    evt.stopPropagation();
    const action = input.dataset.shortcut;
    const code = evt.code;
    if (!code || code === 'Tab') return;
    Object.keys(shortcuts).forEach(key => {
      if (key !== action && shortcuts[key] === code) shortcuts[key] = '';
    });
    shortcuts[action] = code;
    saveShortcuts();
    updateShortcutInputs();
  });
});
document.addEventListener('keydown', evt => {
  const target = evt.target;
  const isEditing = target && (
    target.tagName === 'INPUT' ||
    target.tagName === 'SELECT' ||
    target.tagName === 'TEXTAREA' ||
    target.isContentEditable
  );
  if (isEditing || busy.classList.contains('active')) return;
  if ((evt.ctrlKey || evt.metaKey) && evt.code === 'KeyZ') {
    evt.preventDefault();
    if (evt.shiftKey) redoMask(); else if (!undoMask()) postJson('/undo-result', {}, t('resetUndo'));
    return;
  }
  if ((evt.ctrlKey || evt.metaKey) && evt.code === 'KeyY') {
    evt.preventDefault();
    redoMask();
    return;
  }
  const action = Object.keys(shortcuts).find(key => shortcuts[key] === evt.code);
  if (!action || !shortcutActions[action]) return;
  evt.preventDefault();
  shortcutActions[action]();
});
applyLanguage();
</script>
</body>
</html>"""

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LAMALocal 除尘</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #18211d; background: #f3f5f2; }
body { display: grid; grid-template-rows: auto 1fr; }
.bar { min-height: 58px; display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #fff; border-bottom: 1px solid #d3ddd6; flex-wrap: wrap; }
.brand { font-weight: 700; margin-right: 8px; }
button, select, input[type="file"]::file-selector-button { border: 1px solid #89988f; background: #fff; color: #18211d; border-radius: 6px; padding: 7px 11px; cursor: pointer; }
button.primary { background: #12624f; border-color: #12624f; color: #fff; }
button:disabled, select:disabled, input:disabled { opacity: .55; cursor: wait; }
label { display: inline-flex; align-items: center; gap: 7px; }
input[type="range"] { width: 130px; }
#status { color: #516059; min-width: 220px; }
.workspace { min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 260px; }
.stage { min-height: 0; overflow: auto; display: grid; place-items: start center; padding: 18px; }
.canvasWrap { position: relative; line-height: 0; background: #202522; box-shadow: 0 1px 9px rgba(0,0,0,.18); }
canvas { position: absolute; left: 0; top: 0; }
#imageCanvas { position: static; }
#maskCanvas { opacity: .48; cursor: crosshair; touch-action: none; }
.brushCursor { position: absolute; left: 0; top: 0; width: 0; height: 0; border: 1px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,.82); pointer-events: none; display: none; z-index: 4; transform: translate(-50%, -50%); }
.brushCursor.erase { border-style: dashed; }
.brushCursor { position: absolute; left: 0; top: 0; width: 0; height: 0; border: 1px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,.82); pointer-events: none; display: none; z-index: 4; transform: translate(-50%, -50%); }
.brushCursor.erase { border-style: dashed; }
.empty { margin-top: 14vh; display: grid; gap: 14px; justify-items: center; color: #55635c; }
.empty strong { font-size: 20px; color: #18211d; }
.thumbPanel { border-left: 1px solid #d3ddd6; background: #fff; padding: 14px; display: grid; align-content: start; gap: 10px; }
.thumbTitle { font-size: 13px; font-weight: 700; color: #324139; }
.thumbBox { width: 100%; aspect-ratio: 4 / 3; border: 1px solid #c9d4ce; background: #202522; display: grid; place-items: center; overflow: hidden; }
#thumbCanvas { position: static; max-width: 100%; max-height: 100%; }
.thumbMeta { font-size: 12px; color: #65736c; line-height: 1.4; }
.busy { position: fixed; inset: 0; background: rgba(243,245,242,.78); display: none; place-items: center; z-index: 20; }
.busy.active { display: grid; }
.panel { background: #fff; border: 1px solid #c9d4ce; border-radius: 8px; padding: 20px 24px; display: grid; gap: 12px; justify-items: center; box-shadow: 0 10px 34px rgba(0,0,0,.16); }
.spinner { width: 38px; height: 38px; border: 4px solid #cbd7d0; border-top-color: #12624f; border-radius: 50%; animation: spin .8s linear infinite; }
.progressTrack { width: min(420px, 72vw); height: 10px; border: 1px solid #b7c4bd; border-radius: 999px; overflow: hidden; background: #edf1ee; }
.progressFill { height: 100%; width: 0%; background: #12624f; transition: width .18s ease; }
.progressText { font-size: 13px; color: #516059; min-height: 18px; line-height: 1.4; text-align: center; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (max-width: 860px) {
  .workspace { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }
  .thumbPanel { border-left: 0; border-top: 1px solid #d3ddd6; grid-template-columns: 150px minmax(0, 1fr); align-items: start; }
  .thumbTitle { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<div class="bar">
  <span class="brand">LAMALocal</span>
  <input id="file" type="file" accept="image/*">
  <label>类型 <select id="kind"><option value="color">彩色</option><option value="bw">黑白</option></select></label>
  <button id="auto" class="primary">一键除尘</button>
  <button id="manual" class="primary">手动除尘</button>
  <button id="compare">对比</button>
  <button id="undo">撤销</button>
  <button id="clear">清空标记</button>
  <button id="zoomOut">缩小</button>
  <button id="zoomIn">放大</button>
  <button id="fit">适配</button>
  <label>画笔 <input id="brush" type="range" min="2" max="160" value="34"><span id="brushValue">34</span>px</label>
  <button id="download">下载结果</button>
  <span id="zoomValue">100%</span>
  <span id="status">请先上传图片</span>
</div>
<div class="workspace">
  <div class="stage">
    <div id="empty" class="empty"><strong>上传图片开始除尘</strong><span>可一键自动处理，也可用红色画笔涂抹后手动修复。</span></div>
    <div id="wrap" class="canvasWrap" hidden>
      <canvas id="imageCanvas"></canvas>
      <canvas id="maskCanvas"></canvas>
      <div id="brushCursor" class="brushCursor"></div>
    </div>
  </div>
  <aside class="thumbPanel">
    <div class="thumbTitle">实时缩略图</div>
    <div class="thumbBox"><canvas id="thumbCanvas" width="220" height="165"></canvas></div>
    <div id="thumbMeta" class="thumbMeta">暂无图片</div>
  </aside>
</div>
<div id="busy" class="busy"><div class="panel"><div class="spinner"></div><div id="busyText">正在处理...</div><div class="progressTrack"><div id="progressFill" class="progressFill"></div></div><div id="progressText" class="progressText">0%</div><div id="progressDetail" class="progressText">后台正在准备任务</div></div></div>
<script>
const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const thumbCanvas = document.getElementById('thumbCanvas');
const imageCtx = imageCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const thumbCtx = thumbCanvas.getContext('2d');
const wrap = document.getElementById('wrap');
const empty = document.getElementById('empty');
const statusEl = document.getElementById('status');
const busy = document.getElementById('busy');
const busyText = document.getElementById('busyText');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressDetail = document.getElementById('progressDetail');
const brush = document.getElementById('brush');
const thumbMeta = document.getElementById('thumbMeta');
const controls = [...document.querySelectorAll('button, select, input')];
let drawing = false;
let history = [];
let zoom = 1;
let hasImage = false;
let hasResult = false;
let compareOriginal = false;
let previewScale = 1;
let thumbPending = false;
let progressTimer = null;
let brushPointer = null;
const translations = {
  zh: {
    download: '下载结果', statusUpload: '请先上传图片', sectionProcess: '处理', kind: '类型', color: '彩色', bw: '黑白',
    auto: '一键生成Mask', manual: '点击除尘', paint: '画笔', erase: '橡皮擦', size: '大小', clear: '清空',
    undo: '撤销', showMask: '显示Mask', sectionView: '视图', zoomOut: '缩小', zoomIn: '放大', fit: '适配',
    compare: '对比', settings: '快捷键设置', loadingImage: '正在读取图片...', generatingMask: '正在生成Mask...',
    removingDust: '正在除尘...', done: '处理完成', imageLoaded: '图片已加载', viewOriginal: '正在查看原图',
    viewResult: '正在查看结果', resetUndo: '正在撤销结果...', processing: '正在处理...', space: '空格',
    ttaFlip: '翻转', ttaContrast: '对比度', ttaGamma: 'Gamma', ttaBrightness: '亮度', ttaRotate: '旋转', ttaScale: '缩放',
    ttaMultiscale: '多尺度'
  },
  en: {
    download: 'Download', statusUpload: 'Upload an image first', sectionProcess: 'Process', kind: 'Type',
    color: 'Color', bw: 'B&W', auto: 'Generate Mask', manual: 'Remove Dust', paint: 'Brush', erase: 'Eraser',
    size: 'Size', clear: 'Clear', undo: 'Undo', showMask: 'Show Mask', sectionView: 'View', zoomOut: 'Zoom Out',
    zoomIn: 'Zoom In', fit: 'Fit', compare: 'Compare', settings: 'Shortcut Settings', loadingImage: 'Loading image...',
    generatingMask: 'Generating mask...', removingDust: 'Removing dust...', done: 'Done', imageLoaded: 'Image loaded',
    viewOriginal: 'Viewing original', viewResult: 'Viewing result', resetUndo: 'Undoing result...', processing: 'Processing...',
    space: 'Space', ttaFlip: 'Flip', ttaContrast: 'Contrast', ttaGamma: 'Gamma', ttaBrightness: 'Brightness',
    ttaRotate: 'Rotate', ttaScale: 'Scale', ttaMultiscale: 'Multi-scale'
  }
};
const shortcutDefaults = {
  manual: 'Space',
  zoomIn: 'Equal',
  zoomOut: 'Minus',
  compare: 'KeyC',
  paint: 'KeyB',
  erase: 'KeyE',
};
const shortcutActions = {
  manual: () => runManual(),
  zoomIn: () => setZoom(zoom * 1.25),
  zoomOut: () => setZoom(zoom / 1.25),
  compare: () => toggleCompare(),
  paint: () => setTool('paint'),
  erase: () => setTool('erase'),
};
let language = localStorage.getItem('lamaLanguage') || 'zh';
let storedShortcuts = {};
try {
  storedShortcuts = JSON.parse(localStorage.getItem('lamaShortcuts') || '{}');
} catch (err) {
  storedShortcuts = {};
}
let shortcuts = { ...shortcutDefaults, ...storedShortcuts };

function t(key) { return (translations[language] && translations[language][key]) || translations.zh[key] || key; }
function shortcutText(code) {
  if (code === 'Space') return t('space');
  if (code === 'Equal') return '+';
  if (code === 'Minus') return '-';
  if (code && code.startsWith('Key')) return code.slice(3);
  if (code && code.startsWith('Digit')) return code.slice(5);
  if (code && code.startsWith('Numpad')) return code.replace('Numpad', 'Num ');
  return code || '';
}
function saveShortcuts() {
  localStorage.setItem('lamaShortcuts', JSON.stringify(shortcuts));
}
function updateShortcutInputs() {
  document.querySelectorAll('.shortcutInput').forEach(input => {
    input.value = shortcutText(shortcuts[input.dataset.shortcut]);
  });
}
function applyLanguage() {
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.getElementById('langToggle').textContent = language === 'zh' ? 'English' : '中文';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  updateShortcutInputs();
}

function updateProgress(data) {
  const percent = Math.max(0, Math.min(100, Math.round(data.percent || 0)));
  progressFill.style.width = `${percent}%`;
  progressText.textContent = `${percent}%${data.message ? ' - ' + data.message : ''}`;
  progressDetail.textContent = data.detail || describeProgress(percent, data.message);
  if (data.message) busyText.textContent = data.message;
}
function describeProgress(percent, message) {
  if (message && message.includes('/')) return '后台正在逐块运行 DeepLab 灰尘检测模型';
  if (percent < 10) return '后台正在读取图像、加载模型和准备设备';
  if (percent < 72) return '后台正在对大图滑窗切块并预测灰尘概率';
  if (percent < 82) return '后台正在合并概率图，并做遮罩二值化、闭运算和连通域扩展';
  if (percent < 96) return '后台正在调用 LaMa，只重绘遮罩覆盖的灰尘区域';
  return '后台正在保存结果并生成预览图';
}
async function pollProgress() {
  try {
    const response = await fetch('/progress?t=' + Date.now());
    if (response.ok) updateProgress(await response.json());
  } catch (err) {
  }
}
function startProgress(text) {
  updateProgress({ percent: 0, message: text || t('processing') });
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = setInterval(pollProgress, 500);
  pollProgress();
}
function stopProgress() {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = null;
}

function setBusy(on, text) {
  busy.classList.toggle('active', on);
  busyText.textContent = text || t('processing');
  controls.forEach(el => el.disabled = on);
  if (on) startProgress(text);
  else stopProgress();
}
function applyZoom() {
  wrap.style.transform = `scale(${zoom})`;
  wrap.style.transformOrigin = 'top left';
  wrap.style.marginRight = `${imageCanvas.width * (zoom - 1)}px`;
  wrap.style.marginBottom = `${imageCanvas.height * (zoom - 1)}px`;
  document.getElementById('zoomValue').textContent = `${Math.round(zoom * 100)}%`;
}
function setZoom(next) {
  zoom = Math.max(0.1, Math.min(6, next));
  applyZoom();
}
function fitToStage() {
  const stage = document.querySelector('.stage');
  const zx = (stage.clientWidth - 36) / Math.max(imageCanvas.width, 1);
  const zy = (stage.clientHeight - 36) / Math.max(imageCanvas.height, 1);
  setZoom(Math.min(1, zx, zy));
}
function drawThumbnail() {
  thumbPending = false;
  thumbCtx.clearRect(0, 0, thumbCanvas.width, thumbCanvas.height);
  if (!hasImage) return;
  const scale = Math.min(thumbCanvas.width / imageCanvas.width, thumbCanvas.height / imageCanvas.height);
  const w = Math.max(1, Math.round(imageCanvas.width * scale));
  const h = Math.max(1, Math.round(imageCanvas.height * scale));
  const x = Math.round((thumbCanvas.width - w) / 2);
  const y = Math.round((thumbCanvas.height - h) / 2);
  thumbCtx.drawImage(imageCanvas, x, y, w, h);
  thumbCtx.save();
  thumbCtx.globalAlpha = 0.55;
  thumbCtx.drawImage(maskCanvas, x, y, w, h);
  thumbCtx.restore();
  const originalText = previewScale < 1 ? `原图约 ${Math.round(imageCanvas.width / previewScale)} x ${Math.round(imageCanvas.height / previewScale)}` : `图像 ${imageCanvas.width} x ${imageCanvas.height}`;
  thumbMeta.textContent = `${originalText}，预览 ${imageCanvas.width} x ${imageCanvas.height}`;
}
function scheduleThumbnail() {
  if (thumbPending) return;
  thumbPending = true;
  requestAnimationFrame(drawThumbnail);
}
function pushHistory() {
  if (!hasImage) return;
  history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  if (history.length > 30) history.shift();
}
function point(evt) {
  const r = maskCanvas.getBoundingClientRect();
  return { x: (evt.clientX - r.left) * maskCanvas.width / r.width, y: (evt.clientY - r.top) * maskCanvas.height / r.height };
}
function paint(evt) {
  if (!drawing || !hasImage) return;
  const p = point(evt);
  maskCtx.fillStyle = 'rgba(255,0,0,1)';
  maskCtx.beginPath();
  maskCtx.arc(p.x, p.y, Number(brush.value), 0, Math.PI * 2);
  maskCtx.fill();
  scheduleThumbnail();
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      imageCanvas.width = maskCanvas.width = img.naturalWidth;
      imageCanvas.height = maskCanvas.height = img.naturalHeight;
      imageCtx.drawImage(img, 0, 0);
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      wrap.hidden = false;
      empty.hidden = true;
      hasImage = true;
      history = [];
      redoHistory = [];
      hideBrushCursor();
      fitToStage();
      scheduleThumbnail();
      resolve();
    };
    img.onerror = reject;
    img.src = src;
  });
}
async function postJson(url, payload, text) {
  setBusy(true, text);
  try {
    const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload || {}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || t('done'));
    if (data.preview) {
      hasResult = data.hasResult !== false;
      compareOriginal = false;
      if (data.previewScale) previewScale = data.previewScale;
      await loadImage(data.preview + '?t=' + Date.now());
    }
    statusEl.textContent = data.message || t('done');
    return data;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    setBusy(false);
  }
}
async function postForm(url, formData, text) {
  setBusy(true, text);
  try {
    const response = await fetch(url, { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || t('statusUpload'));
    if (data.preview) {
      hasResult = data.hasResult !== false;
      compareOriginal = false;
      previewScale = data.previewScale || 1;
      await loadImage(data.preview + '?t=' + Date.now());
    }
    statusEl.textContent = data.message || t('imageLoaded');
    return data;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    setBusy(false);
  }
}
document.getElementById('file').onchange = async evt => {
  const file = evt.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('image', file, file.name);
  await postForm('/upload', formData, '正在读取图片...');
};
brush.oninput = () => document.getElementById('brushValue').textContent = brush.value;
maskCanvas.addEventListener('pointerdown', evt => { drawing = true; pushHistory(); maskCanvas.setPointerCapture(evt.pointerId); paint(evt); });
maskCanvas.addEventListener('pointermove', paint);
maskCanvas.addEventListener('pointerup', () => { drawing = false; scheduleThumbnail(); });
maskCanvas.addEventListener('pointercancel', () => drawing = false);
document.getElementById('auto').onclick = () => postJson('/auto', { kind: document.getElementById('kind').value }, '正在一键除尘...');
document.getElementById('manual').onclick = () => postJson('/manual', { mask: maskCanvas.toDataURL('image/png') }, '正在手动除尘...');
document.getElementById('undo').onclick = () => {
  const last = history.pop();
  if (last) { maskCtx.putImageData(last, 0, 0); scheduleThumbnail(); return; }
  postJson('/undo-result', {}, '正在撤销结果...');
};
document.getElementById('clear').onclick = () => { pushHistory(); maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); scheduleThumbnail(); };
document.getElementById('zoomIn').onclick = () => setZoom(zoom * 1.25);
document.getElementById('zoomOut').onclick = () => setZoom(zoom / 1.25);
document.getElementById('fit').onclick = () => {
  fitToStage();
};
document.getElementById('compare').onclick = async () => {
  if (!hasResult) return;
  compareOriginal = !compareOriginal;
  await loadImage((compareOriginal ? '/image' : '/result') + '?t=' + Date.now());
  statusEl.textContent = compareOriginal ? '正在查看原图' : '正在查看结果';
};
document.getElementById('download').onclick = () => {
  if (hasResult) window.location.href = '/download?t=' + Date.now();
};
document.getElementById('langToggle').onclick = () => {
  language = language === 'zh' ? 'en' : 'zh';
  localStorage.setItem('lamaLanguage', language);
  applyLanguage();
};
document.querySelectorAll('.shortcutInput').forEach(input => {
  input.addEventListener('keydown', evt => {
    evt.preventDefault();
    evt.stopPropagation();
    const action = input.dataset.shortcut;
    const code = evt.code;
    if (!code || code === 'Tab') return;
    Object.keys(shortcuts).forEach(key => {
      if (key !== action && shortcuts[key] === code) shortcuts[key] = '';
    });
    shortcuts[action] = code;
    saveShortcuts();
    updateShortcutInputs();
  });
});
document.addEventListener('keydown', evt => {
  const target = evt.target;
  const isEditing = target && (
    target.tagName === 'INPUT' ||
    target.tagName === 'SELECT' ||
    target.tagName === 'TEXTAREA' ||
    target.isContentEditable
  );
  if (isEditing || busy.classList.contains('active')) return;
  if ((evt.ctrlKey || evt.metaKey) && evt.code === 'KeyZ') {
    evt.preventDefault();
    if (evt.shiftKey) redoMask(); else if (!undoMask()) postJson('/undo-result', {}, t('resetUndo'));
    return;
  }
  if ((evt.ctrlKey || evt.metaKey) && evt.code === 'KeyY') {
    evt.preventDefault();
    redoMask();
    return;
  }
  const action = Object.keys(shortcuts).find(key => shortcuts[key] === evt.code);
  if (!action || !shortcutActions[action]) return;
  evt.preventDefault();
  shortcutActions[action]();
});
applyLanguage();
</script>
</body>
</html>"""

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LAMALocal 除尘</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #18211d; background: #e9eeeb; }
body { display: grid; grid-template-rows: auto 1fr; }
.bar { min-height: 56px; display: flex; align-items: center; gap: 14px; padding: 10px 16px; background: #fdfefd; border-bottom: 1px solid #cfd8d2; }
.brand { font-weight: 750; font-size: 17px; }
.spacer { flex: 1; }
button, select, input[type="file"]::file-selector-button { border: 1px solid #93a29a; background: #fff; color: #18211d; border-radius: 6px; padding: 7px 11px; cursor: pointer; }
button.primary { background: #12624f; border-color: #12624f; color: #fff; }
button.tool.active { background: #dceee8; border-color: #12624f; color: #0d4f40; }
button.compact { padding: 6px 9px; font-size: 12px; }
button:disabled, select:disabled, input:disabled { opacity: .55; cursor: wait; }
label { display: inline-flex; align-items: center; gap: 7px; }
input[type="range"] { width: 100%; }
#status { color: #516059; min-width: 220px; text-align: right; }
.workspace { min-height: 0; display: grid; grid-template-columns: 286px minmax(0, 1fr) 260px; }
.sidePanel { min-height: 0; overflow: auto; border-right: 1px solid #cfd8d2; background: #fbfcfb; padding: 14px; display: grid; align-content: start; gap: 14px; }
.section { display: grid; gap: 10px; padding-bottom: 14px; border-bottom: 1px solid #d9e1dc; }
.section:last-child { border-bottom: 0; }
.sectionTitle { font-size: 12px; font-weight: 750; color: #4b5a52; text-transform: uppercase; letter-spacing: .04em; }
.field { display: grid; gap: 5px; font-size: 12px; color: #415047; }
.field input, .field select, .advancedPanel input, .advancedPanel select { width: 100%; min-width: 0; border: 1px solid #a9b6af; border-radius: 6px; padding: 7px 8px; background: #fff; color: #18211d; }
.check { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #26352e; }
.buttonGrid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.buttonRow { display: flex; gap: 8px; flex-wrap: wrap; }
.buttonRow button { flex: 1; }
.advanced summary { list-style: none; border: 1px solid #93a29a; background: #fff; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
.advanced summary::-webkit-details-marker { display: none; }
.advancedPanel { display: grid; gap: 10px; margin-top: 10px; }
.advancedPanel label { display: grid; gap: 5px; font-size: 12px; color: #415047; }
.ttaOptions { display: grid; gap: 4px; max-width: 100%; overflow: visible; }
.ttaRow { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 8px; width: 100%; min-height: 22px; color: #415047; font-size: 12px; line-height: 1.2; }
.ttaRow span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ttaRow input { width: 14px; height: 14px; margin: 0; justify-self: end; }
.ttaMaster { font-size: 13px; font-weight: 600; color: #26352e; }
.ttaGrid { display: grid; grid-template-columns: 1fr; gap: 3px; max-width: 100%; }
.stage { min-height: 0; overflow: auto; display: grid; place-items: start center; padding: 22px; background: #dfe5e1; }
.canvasWrap { position: relative; line-height: 0; background: #202522; box-shadow: 0 1px 10px rgba(0,0,0,.22); }
canvas { position: absolute; left: 0; top: 0; }
#imageCanvas { position: static; }
#maskCanvas { opacity: .48; cursor: crosshair; touch-action: none; }
.brushCursor { position: absolute; left: 0; top: 0; width: 0; height: 0; border: 1px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,.82); pointer-events: none; display: none; z-index: 4; transform: translate(-50%, -50%); }
.brushCursor.erase { border-style: dashed; }
.empty { margin-top: 14vh; display: grid; gap: 14px; justify-items: center; color: #55635c; }
.empty strong { font-size: 20px; color: #18211d; }
.thumbPanel { border-left: 1px solid #d3ddd6; background: #fff; padding: 14px; display: grid; align-content: start; gap: 10px; }
.thumbTitle { font-size: 13px; font-weight: 700; color: #324139; }
.thumbBox { width: 100%; aspect-ratio: 4 / 3; border: 1px solid #c9d4ce; background: #202522; display: grid; place-items: center; overflow: hidden; }
#thumbCanvas { position: static; max-width: 100%; max-height: 100%; }
.thumbMeta { font-size: 12px; color: #65736c; line-height: 1.4; }
.busy { position: fixed; inset: 0; background: rgba(243,245,242,.78); display: none; place-items: center; z-index: 20; }
.busy.active { display: grid; }
.panel { background: #fff; border: 1px solid #c9d4ce; border-radius: 8px; padding: 20px 24px; display: grid; gap: 12px; justify-items: center; box-shadow: 0 10px 34px rgba(0,0,0,.16); }
.spinner { width: 38px; height: 38px; border: 4px solid #cbd7d0; border-top-color: #12624f; border-radius: 50%; animation: spin .8s linear infinite; }
.progressTrack { width: min(420px, 72vw); height: 10px; border: 1px solid #b7c4bd; border-radius: 999px; overflow: hidden; background: #edf1ee; }
.progressFill { height: 100%; width: 0%; background: #12624f; transition: width .18s ease; }
.progressText { font-size: 13px; color: #516059; min-height: 18px; line-height: 1.4; text-align: center; }
.shortcutSettings { position: fixed; right: 12px; bottom: 12px; width: 230px; z-index: 12; font-size: 12px; }
.shortcutSettings details { border: 1px solid #c7d2cc; border-radius: 8px; background: rgba(255,255,255,.96); box-shadow: 0 6px 22px rgba(0,0,0,.12); }
.shortcutSettings summary { list-style: none; padding: 8px 10px; cursor: pointer; font-weight: 700; color: #26352e; }
.shortcutSettings summary::-webkit-details-marker { display: none; }
.shortcutPanel { display: grid; gap: 8px; padding: 0 10px 10px; }
.shortcutRow { display: grid; grid-template-columns: minmax(0, 1fr) 72px; align-items: center; gap: 8px; }
.shortcutInput { width: 72px; text-align: center; border: 1px solid #a9b6af; border-radius: 6px; padding: 6px 4px; background: #fff; color: #18211d; }
.shortcutInput:focus { outline: 2px solid #8cc0b1; border-color: #12624f; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (max-width: 980px) {
  .workspace { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr) auto; }
  .sidePanel { border-right: 0; border-bottom: 1px solid #cfd8d2; grid-template-columns: repeat(2, minmax(220px, 1fr)); align-items: start; }
  .thumbPanel { border-left: 0; border-top: 1px solid #d3ddd6; grid-template-columns: 150px minmax(0, 1fr); align-items: start; }
  .thumbTitle { grid-column: 1 / -1; }
}
@media (max-width: 620px) {
  .bar { flex-wrap: wrap; }
  .sidePanel { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="bar">
  <span class="brand">LAMALocal</span>
  <input id="file" type="file" accept="image/*">
  <div class="spacer"></div>
  <button id="langToggle" class="compact">English</button>
  <button id="download" data-i18n="download">下载结果</button>
  <span id="zoomValue">100%</span>
  <span id="status" data-i18n="statusUpload">请先上传图片</span>
</div>
<div class="workspace">
  <aside class="sidePanel">
    <section class="section">
      <div class="sectionTitle" data-i18n="sectionProcess">处理</div>
      <label class="field"><span data-i18n="kind">类型</span> <select id="kind"><option value="color" data-i18n="color">彩色</option><option value="bw" data-i18n="bw">黑白</option></select></label>
      <button id="auto" class="primary" data-i18n="auto">一键生成Mask</button>
      <button id="manual" class="primary" data-i18n="manual">点击除尘</button>
    </section>
    <section class="section">
      <div class="sectionTitle">Mask</div>
      <div class="buttonGrid">
        <button id="paintMode" class="tool active" data-i18n="paint">画笔</button>
        <button id="eraseMode" class="tool" data-i18n="erase">橡皮擦</button>
      </div>
      <label class="field"><span data-i18n="size">大小</span> <input id="brush" type="range" min="2" max="160" value="34"></label>
      <div class="buttonRow">
        <button id="clear" data-i18n="clear">清空</button>
        <button id="undo" data-i18n="undo">撤销</button>
      </div>
      <label class="check"><input id="maskVisible" type="checkbox" checked> <span data-i18n="showMask">显示Mask</span></label>
    </section>
    <section class="section">
      <div class="sectionTitle" data-i18n="sectionView">视图</div>
      <div class="buttonGrid">
        <button id="zoomOut" data-i18n="zoomOut">缩小</button>
        <button id="zoomIn" data-i18n="zoomIn">放大</button>
      </div>
      <div class="buttonRow">
        <button id="fit" data-i18n="fit">适配</button>
        <button id="compare" data-i18n="compare">对比</button>
      </div>
    </section>
    <section class="section">
      <div class="sectionTitle">高级</div>
      <details class="advanced">
        <summary>参数</summary>
        <div class="advancedPanel">
          <label>增强方式 <select id="maskEnhance"><option value="none" selected>无</option><option value="sharpen">锐化</option></select></label>
          <label>阈值 <input id="threshold" type="number" min="0.01" max="0.99" step="0.01" value="0.18"></label>
          <label>输入尺寸 <input id="imageSize" type="number" min="128" max="2048" step="32" value="512"></label>
          <label>分块大小 <input id="tileSize" type="number" min="0" max="2048" step="32" value="512"></label>
          <label>重叠像素 <input id="tileOverlap" type="number" min="0" max="1024" step="16" value="128"></label>
          <label>闭运算半径 <input id="close" type="number" min="0" max="64" step="1" value="5"></label>
          <label>组件扩张 <input id="componentExpand" type="number" min="0" max="128" step="1" value="20"></label>
          <label>全局膨胀 <input id="dilate" type="number" min="0" max="64" step="1" value="0"></label>
          <div class="ttaOptions">
            <label class="ttaRow ttaMaster"><span>TTA</span><input id="tta" type="checkbox" checked></label>
            <div class="ttaGrid">
              <label class="ttaRow"><span data-i18n="ttaFlip">翻转</span><input class="ttaType" type="checkbox" value="flip" checked></label>
              <label class="ttaRow"><span data-i18n="ttaContrast">对比度</span><input class="ttaType" type="checkbox" value="contrast"></label>
              <label class="ttaRow"><span data-i18n="ttaGamma">Gamma</span><input class="ttaType" type="checkbox" value="gamma"></label>
              <label class="ttaRow"><span data-i18n="ttaBrightness">亮度</span><input class="ttaType" type="checkbox" value="brightness"></label>
              <label class="ttaRow"><span data-i18n="ttaRotate">旋转</span><input class="ttaType" type="checkbox" value="rotate"></label>
              <label class="ttaRow"><span data-i18n="ttaScale">缩放</span><input class="ttaType" type="checkbox" value="scale"></label>
              <label class="ttaRow"><span data-i18n="ttaMultiscale">多尺度</span><input class="ttaType" type="checkbox" value="multiscale"></label>
            </div>
          </div>
        </div>
      </details>
    </section>
  </aside>
  <div class="stage">
    <div id="empty" class="empty"><strong>上传图片开始除尘</strong><span>先生成 Mask，涂改后点击除尘。</span></div>
    <div id="wrap" class="canvasWrap" hidden>
      <canvas id="imageCanvas"></canvas>
      <canvas id="maskCanvas"></canvas>
      <div id="brushCursor" class="brushCursor"></div>
    </div>
  </div>
  <aside class="thumbPanel">
    <div class="thumbTitle">实时缩略图</div>
    <div class="thumbBox"><canvas id="thumbCanvas" width="220" height="165"></canvas></div>
    <div id="thumbMeta" class="thumbMeta">暂无图片</div>
  </aside>
</div>
<div class="shortcutSettings">
  <details id="shortcutDetails">
    <summary data-i18n="settings">快捷键设置</summary>
    <div class="shortcutPanel">
      <label class="shortcutRow"><span data-i18n="manual">点击除尘</span><input class="shortcutInput" data-shortcut="manual" readonly></label>
      <label class="shortcutRow"><span data-i18n="zoomIn">放大</span><input class="shortcutInput" data-shortcut="zoomIn" readonly></label>
      <label class="shortcutRow"><span data-i18n="zoomOut">缩小</span><input class="shortcutInput" data-shortcut="zoomOut" readonly></label>
      <label class="shortcutRow"><span data-i18n="compare">对比</span><input class="shortcutInput" data-shortcut="compare" readonly></label>
      <label class="shortcutRow"><span data-i18n="paint">画笔</span><input class="shortcutInput" data-shortcut="paint" readonly></label>
      <label class="shortcutRow"><span data-i18n="erase">橡皮擦</span><input class="shortcutInput" data-shortcut="erase" readonly></label>
    </div>
  </details>
</div>
<div id="busy" class="busy"><div class="panel"><div class="spinner"></div><div id="busyText">正在处理...</div><div class="progressTrack"><div id="progressFill" class="progressFill"></div></div><div id="progressText" class="progressText">0%</div><div id="progressDetail" class="progressText">后台正在准备任务</div></div></div>
<script>
const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const thumbCanvas = document.getElementById('thumbCanvas');
const imageCtx = imageCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const thumbCtx = thumbCanvas.getContext('2d');
const wrap = document.getElementById('wrap');
const empty = document.getElementById('empty');
const statusEl = document.getElementById('status');
const busy = document.getElementById('busy');
const busyText = document.getElementById('busyText');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressDetail = document.getElementById('progressDetail');
const brush = document.getElementById('brush');
const paintMode = document.getElementById('paintMode');
const eraseMode = document.getElementById('eraseMode');
const maskVisible = document.getElementById('maskVisible');
const brushCursor = document.getElementById('brushCursor');
const thumbMeta = document.getElementById('thumbMeta');
const controls = [...document.querySelectorAll('button, select, input')];
let drawing = false;
let history = [];
let redoHistory = [];
let zoom = 1;
let toolMode = 'paint';
let maskLayerVisible = true;
let hasImage = false;
let hasResult = false;
let compareOriginal = false;
let previewScale = 1;
let thumbPending = false;
let progressTimer = null;
const translations = {
  zh: {
    download: '下载结果', statusUpload: '请先上传图片', sectionProcess: '处理', kind: '类型', color: '彩色', bw: '黑白',
    auto: '一键生成Mask', manual: '点击除尘', paint: '画笔', erase: '橡皮擦', size: '大小', clear: '清空',
    undo: '撤销', showMask: '显示Mask', sectionView: '视图', zoomOut: '缩小', zoomIn: '放大', fit: '适配',
    compare: '对比', settings: '快捷键设置', loadingImage: '正在读取图片...', generatingMask: '正在生成Mask...',
    removingDust: '正在除尘...', done: '处理完成', imageLoaded: '图片已加载', viewOriginal: '正在查看原图',
    viewResult: '正在查看结果', resetUndo: '正在撤销结果...', processing: '正在处理...', space: '空格',
    ttaFlip: '翻转', ttaContrast: '对比度', ttaGamma: 'Gamma', ttaBrightness: '亮度', ttaRotate: '旋转', ttaScale: '缩放'
  },
  en: {
    download: 'Download', statusUpload: 'Upload an image first', sectionProcess: 'Process', kind: 'Type',
    color: 'Color', bw: 'B&W', auto: 'Generate Mask', manual: 'Remove Dust', paint: 'Brush', erase: 'Eraser',
    size: 'Size', clear: 'Clear', undo: 'Undo', showMask: 'Show Mask', sectionView: 'View', zoomOut: 'Zoom Out',
    zoomIn: 'Zoom In', fit: 'Fit', compare: 'Compare', settings: 'Shortcut Settings', loadingImage: 'Loading image...',
    generatingMask: 'Generating mask...', removingDust: 'Removing dust...', done: 'Done', imageLoaded: 'Image loaded',
    viewOriginal: 'Viewing original', viewResult: 'Viewing result', resetUndo: 'Undoing result...', processing: 'Processing...',
    space: 'Space', ttaFlip: 'Flip', ttaContrast: 'Contrast', ttaGamma: 'Gamma', ttaBrightness: 'Brightness',
    ttaRotate: 'Rotate', ttaScale: 'Scale'
  }
};
const shortcutDefaults = {
  manual: 'Space',
  zoomIn: 'Equal',
  zoomOut: 'Minus',
  compare: 'KeyC',
  paint: 'KeyB',
  erase: 'KeyE',
};
const shortcutActions = {
  manual: () => runManual(),
  zoomIn: () => setZoom(zoom * 1.25),
  zoomOut: () => setZoom(zoom / 1.25),
  compare: () => toggleCompare(),
  paint: () => setTool('paint'),
  erase: () => setTool('erase'),
};
let language = localStorage.getItem('lamaLanguage') || 'zh';
let storedShortcuts = {};
try {
  storedShortcuts = JSON.parse(localStorage.getItem('lamaShortcuts') || '{}');
} catch (err) {
  storedShortcuts = {};
}
let shortcuts = { ...shortcutDefaults, ...storedShortcuts };

function t(key) { return (translations[language] && translations[language][key]) || translations.zh[key] || key; }
function shortcutText(code) {
  if (code === 'Space') return t('space');
  if (code === 'Equal') return '+';
  if (code === 'Minus') return '-';
  if (code && code.startsWith('Key')) return code.slice(3);
  if (code && code.startsWith('Digit')) return code.slice(5);
  if (code && code.startsWith('Numpad')) return code.replace('Numpad', 'Num ');
  return code || '';
}
function saveShortcuts() {
  localStorage.setItem('lamaShortcuts', JSON.stringify(shortcuts));
}
function updateShortcutInputs() {
  document.querySelectorAll('.shortcutInput').forEach(input => {
    input.value = shortcutText(shortcuts[input.dataset.shortcut]);
  });
}
function applyLanguage() {
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.getElementById('langToggle').textContent = language === 'zh' ? 'English' : '中文';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  updateShortcutInputs();
}

function numericOption(id) { return Number(document.getElementById(id).value); }
function advancedOptions() {
  return {
    mask_enhance: document.getElementById('maskEnhance').value,
    threshold: numericOption('threshold'),
    image_size: numericOption('imageSize'),
    tile_size: numericOption('tileSize'),
    tile_overlap: numericOption('tileOverlap'),
    close: numericOption('close'),
    component_expand: numericOption('componentExpand'),
    dilate: numericOption('dilate'),
    tta: document.getElementById('tta').checked,
    tta_types: [...document.querySelectorAll('.ttaType:checked')].map(input => input.value),
  };
}
function describeProgress(percent, message) {
  if (message && message.includes('/')) return '后台正在逐块运行 DeepLab 灰尘检测模型';
  if (percent < 10) return '后台正在读取图像、加载模型和准备设备';
  if (percent < 72) return '后台正在对大图滑窗切块并预测灰尘概率';
  if (percent < 82) return '后台正在合并概率图并生成遮罩';
  if (percent < 96) return '后台正在调用 LaMa 修复遮罩覆盖区域';
  return '后台正在保存结果并生成预览图';
}
function updateProgress(data) {
  const percent = Math.max(0, Math.min(100, Math.round(data.percent || 0)));
  progressFill.style.width = `${percent}%`;
  progressText.textContent = `${percent}%${data.message ? ' - ' + data.message : ''}`;
  progressDetail.textContent = data.detail || describeProgress(percent, data.message);
  if (data.message) busyText.textContent = data.message;
}
async function pollProgress() {
  try {
    const response = await fetch('/progress?t=' + Date.now());
    if (response.ok) updateProgress(await response.json());
  } catch (err) {}
}
function startProgress(text) {
  updateProgress({ percent: 0, message: text || '正在处理...' });
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = setInterval(pollProgress, 500);
  pollProgress();
}
function stopProgress() {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = null;
}
function setBusy(on, text) {
  busy.classList.toggle('active', on);
  busyText.textContent = text || '正在处理...';
  controls.forEach(el => el.disabled = on);
  if (on) startProgress(text); else stopProgress();
}
function applyZoom() {
  wrap.style.transform = `scale(${zoom})`;
  wrap.style.transformOrigin = 'top left';
  wrap.style.marginRight = `${imageCanvas.width * (zoom - 1)}px`;
  wrap.style.marginBottom = `${imageCanvas.height * (zoom - 1)}px`;
  document.getElementById('zoomValue').textContent = `${Math.round(zoom * 100)}%`;
}
function setZoom(next) {
  zoom = Math.max(0.1, Math.min(6, next));
  applyZoom();
}
function fitToStage() {
  const stage = document.querySelector('.stage');
  const zx = (stage.clientWidth - 36) / Math.max(imageCanvas.width, 1);
  const zy = (stage.clientHeight - 36) / Math.max(imageCanvas.height, 1);
  setZoom(Math.min(1, zx, zy));
}
function setTool(mode) {
  toolMode = mode;
  paintMode.classList.toggle('active', mode === 'paint');
  eraseMode.classList.toggle('active', mode === 'erase');
  maskCanvas.style.cursor = 'none';
  renderBrushCursor();
}
function setMaskVisible(visible) {
  maskLayerVisible = visible;
  maskCanvas.style.opacity = visible ? '.48' : '0';
  scheduleThumbnail();
}
function drawThumbnail() {
  thumbPending = false;
  thumbCtx.clearRect(0, 0, thumbCanvas.width, thumbCanvas.height);
  if (!hasImage) return;
  const scale = Math.min(thumbCanvas.width / imageCanvas.width, thumbCanvas.height / imageCanvas.height);
  const w = Math.max(1, Math.round(imageCanvas.width * scale));
  const h = Math.max(1, Math.round(imageCanvas.height * scale));
  const x = Math.round((thumbCanvas.width - w) / 2);
  const y = Math.round((thumbCanvas.height - h) / 2);
  thumbCtx.drawImage(imageCanvas, x, y, w, h);
  if (maskLayerVisible) {
    thumbCtx.save();
    thumbCtx.globalAlpha = 0.55;
    thumbCtx.drawImage(maskCanvas, x, y, w, h);
    thumbCtx.restore();
  }
  const originalText = previewScale < 1 ? `原图约 ${Math.round(imageCanvas.width / previewScale)} x ${Math.round(imageCanvas.height / previewScale)}` : `图像 ${imageCanvas.width} x ${imageCanvas.height}`;
  thumbMeta.textContent = `${originalText}，预览 ${imageCanvas.width} x ${imageCanvas.height}`;
}
function scheduleThumbnail() {
  if (thumbPending) return;
  thumbPending = true;
  requestAnimationFrame(drawThumbnail);
}
function pushHistory() {
  if (!hasImage) return;
  history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  if (history.length > 30) history.shift();
  redoHistory = [];
}
function captureMask() {
  return maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
}
function restoreMask(state) {
  maskCtx.putImageData(state, 0, 0);
  scheduleThumbnail();
}
function undoMask() {
  if (!history.length) return false;
  redoHistory.push(captureMask());
  restoreMask(history.pop());
  return true;
}
function redoMask() {
  if (!redoHistory.length || !hasImage) return false;
  history.push(captureMask());
  if (history.length > 30) history.shift();
  restoreMask(redoHistory.pop());
  return true;
}
document.addEventListener('keydown', evt => {
  const target = evt.target;
  const isEditing = target && (
    target.tagName === 'INPUT' ||
    target.tagName === 'SELECT' ||
    target.tagName === 'TEXTAREA' ||
    target.isContentEditable
  );
  if (isEditing || busy.classList.contains('active') || !(evt.ctrlKey || evt.metaKey)) return;
  if (evt.code === 'KeyZ') {
    evt.preventDefault();
    evt.stopImmediatePropagation();
    if (evt.shiftKey) redoMask(); else if (!undoMask()) postJson('/undo-result', {}, t('resetUndo'));
  } else if (evt.code === 'KeyY') {
    evt.preventDefault();
    evt.stopImmediatePropagation();
    redoMask();
  }
}, true);
function point(evt) {
  const r = maskCanvas.getBoundingClientRect();
  return { x: (evt.clientX - r.left) * maskCanvas.width / r.width, y: (evt.clientY - r.top) * maskCanvas.height / r.height };
}
function renderBrushCursor() {
  if (!hasImage || !brushPointer) {
    brushCursor.style.display = 'none';
    return;
  }
  const radius = Number(brush.value);
  brushCursor.style.display = 'block';
  brushCursor.style.width = `${radius * 2}px`;
  brushCursor.style.height = `${radius * 2}px`;
  brushCursor.style.left = `${brushPointer.x}px`;
  brushCursor.style.top = `${brushPointer.y}px`;
  brushCursor.classList.toggle('erase', toolMode === 'erase');
}
function updateBrushCursor(evt) {
  brushPointer = point(evt);
  renderBrushCursor();
}
function hideBrushCursor() {
  brushPointer = null;
  brushCursor.style.display = 'none';
}
function adjustBrushSize(delta) {
  const current = Number(brush.value);
  const min = Number(brush.min || 1);
  const max = Number(brush.max || 300);
  const step = current >= 80 ? 6 : current >= 30 ? 4 : 2;
  const next = Math.max(min, Math.min(max, current + delta * step));
  brush.value = String(next);
  brush.dispatchEvent(new Event('input', { bubbles: true }));
}
function paint(evt) {
  if (!drawing || !hasImage) return;
  const p = point(evt);
  maskCtx.save();
  maskCtx.globalCompositeOperation = toolMode === 'erase' ? 'destination-out' : 'source-over';
  maskCtx.fillStyle = 'rgba(255,0,0,1)';
  maskCtx.beginPath();
  maskCtx.arc(p.x, p.y, Number(brush.value), 0, Math.PI * 2);
  maskCtx.fill();
  maskCtx.restore();
  scheduleThumbnail();
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      imageCanvas.width = maskCanvas.width = img.naturalWidth;
      imageCanvas.height = maskCanvas.height = img.naturalHeight;
      imageCtx.drawImage(img, 0, 0);
      maskCtx.globalCompositeOperation = 'source-over';
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      wrap.hidden = false;
      empty.hidden = true;
      hasImage = true;
      fitToStage();
      scheduleThumbnail();
      resolve();
    };
    img.onerror = reject;
    img.src = src;
  });
}
function loadMask(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      const buffer = document.createElement('canvas');
      buffer.width = maskCanvas.width;
      buffer.height = maskCanvas.height;
      const ctx = buffer.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, buffer.width, buffer.height);
      const data = ctx.getImageData(0, 0, buffer.width, buffer.height);
      for (let i = 0; i < data.data.length; i += 4) {
        const v = Math.max(data.data[i], data.data[i + 1], data.data[i + 2]);
        data.data[i] = 255;
        data.data[i + 1] = 0;
        data.data[i + 2] = 0;
        data.data[i + 3] = v >= 128 ? 255 : 0;
      }
      maskCtx.putImageData(data, 0, 0);
      history = [];
      redoHistory = [];
      maskVisible.checked = true;
      setMaskVisible(true);
      scheduleThumbnail();
      resolve();
    };
    img.onerror = reject;
    img.src = src;
  });
}
async function postJson(url, payload, text) {
  setBusy(true, text);
  try {
    const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload || {}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '处理失败');
    if (data.preview) {
      hasResult = data.hasResult !== false;
      compareOriginal = false;
      if (data.previewScale) previewScale = data.previewScale;
      await loadImage(data.preview + '?t=' + Date.now());
    }
    if (data.mask) await loadMask(data.mask + '?t=' + Date.now());
    statusEl.textContent = data.message || '处理完成';
    return data;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    setBusy(false);
  }
}
async function postForm(url, formData, text) {
  setBusy(true, text);
  try {
    const response = await fetch(url, { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '上传失败');
    if (data.preview) {
      hasResult = data.hasResult !== false;
      compareOriginal = false;
      previewScale = data.previewScale || 1;
      await loadImage(data.preview + '?t=' + Date.now());
    }
    statusEl.textContent = data.message || '图片已加载';
    return data;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    setBusy(false);
  }
}
document.getElementById('file').onchange = async evt => {
  const file = evt.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('image', file, file.name);
  await postForm('/upload', formData, t('loadingImage'));
};
paintMode.onclick = () => setTool('paint');
eraseMode.onclick = () => setTool('erase');
maskVisible.onchange = () => setMaskVisible(maskVisible.checked);
brush.oninput = () => renderBrushCursor();
maskCanvas.addEventListener('pointerenter', updateBrushCursor);
maskCanvas.addEventListener('pointermove', evt => { updateBrushCursor(evt); paint(evt); });
maskCanvas.addEventListener('pointerleave', evt => { if (!drawing) hideBrushCursor(); });
maskCanvas.addEventListener('pointerdown', evt => { updateBrushCursor(evt); drawing = true; pushHistory(); maskCanvas.setPointerCapture(evt.pointerId); paint(evt); });
maskCanvas.addEventListener('pointerup', evt => { drawing = false; updateBrushCursor(evt); scheduleThumbnail(); });
maskCanvas.addEventListener('pointercancel', () => { drawing = false; hideBrushCursor(); });
maskCanvas.addEventListener('wheel', evt => {
  if (!hasImage) return;
  if (evt.altKey) {
    evt.preventDefault();
    updateBrushCursor(evt);
    adjustBrushSize(evt.deltaY < 0 ? 1 : -1);
    return;
  }
  if (evt.ctrlKey || evt.metaKey) {
    evt.preventDefault();
    updateBrushCursor(evt);
    setZoom(zoom * (evt.deltaY < 0 ? 1.12 : 1 / 1.12));
  }
}, { passive: false });
function runAuto() {
  return postJson('/auto', {
  kind: document.getElementById('kind').value,
  options: advancedOptions()
  }, t('generatingMask'));
}
function runManual() {
  return postJson('/manual', { mask: maskCanvas.toDataURL('image/png') }, t('removingDust'));
}
document.getElementById('auto').onclick = runAuto;
document.getElementById('manual').onclick = runManual;
document.getElementById('undo').onclick = () => {
  if (undoMask()) return;
  postJson('/undo-result', {}, t('resetUndo'));
};
document.getElementById('clear').onclick = () => { pushHistory(); maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); scheduleThumbnail(); };
document.getElementById('zoomIn').onclick = () => setZoom(zoom * 1.25);
document.getElementById('zoomOut').onclick = () => setZoom(zoom / 1.25);
document.getElementById('fit').onclick = () => {
  fitToStage();
};
async function toggleCompare() {
  if (!hasResult) return;
  compareOriginal = !compareOriginal;
  await loadImage((compareOriginal ? '/image' : '/result') + '?t=' + Date.now());
  statusEl.textContent = compareOriginal ? t('viewOriginal') : t('viewResult');
}
document.getElementById('compare').onclick = toggleCompare;
document.getElementById('download').onclick = () => {
  if (hasResult) window.location.href = '/download?t=' + Date.now();
};
document.getElementById('langToggle').onclick = () => {
  language = language === 'zh' ? 'en' : 'zh';
  localStorage.setItem('lamaLanguage', language);
  applyLanguage();
};
document.querySelectorAll('.shortcutInput').forEach(input => {
  input.addEventListener('keydown', evt => {
    evt.preventDefault();
    evt.stopPropagation();
    const action = input.dataset.shortcut;
    const code = evt.code;
    if (!code || code === 'Tab') return;
    Object.keys(shortcuts).forEach(key => {
      if (key !== action && shortcuts[key] === code) shortcuts[key] = '';
    });
    shortcuts[action] = code;
    saveShortcuts();
    updateShortcutInputs();
  });
});
document.addEventListener('keydown', evt => {
  const target = evt.target;
  const isEditing = target && (
    target.tagName === 'INPUT' ||
    target.tagName === 'SELECT' ||
    target.tagName === 'TEXTAREA' ||
    target.isContentEditable
  );
  if (isEditing || busy.classList.contains('active')) return;
  const action = Object.keys(shortcuts).find(key => shortcuts[key] === evt.code);
  if (!action || !shortcutActions[action]) return;
  evt.preventDefault();
  shortcutActions[action]();
});
applyLanguage();
</script>
</body>
</html>"""


def app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def data_path(relative):
    if hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / relative
        if bundled.exists():
            return bundled
    return app_root() / relative


def decode_data_url(data_url):
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("Only image files are supported")
    return base64.b64decode(encoded)


class DustApp:
    def __init__(self, args):
        self.args = args
        self.work_dir = Path(args.work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = None
        self.result_path = None
        self.mask_path = None
        self.preview_path = None
        self.result_preview_path = None
        self.source_format = None
        self.source_save_info = {}
        self.preview_scale = 1.0
        self.history = []
        self._device = None
        self._deeplab = {}
        self._lama = None
        self._progress_lock = threading.Lock()
        self._progress = {"active": False, "percent": 0, "message": "空闲", "detail": ""}

        root = app_root()
        scripts_dir = root / "scripts"
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(scripts_dir))

    def infer(self):
        import infer_deeplab_lama as infer

        return infer

    def device(self):
        if self._device is None:
            import torch

            if self.args.device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.args.device)
        return self._device

    def set_progress(self, percent, message=None, active=True, detail=None):
        with self._progress_lock:
            self._progress = {
                "active": bool(active),
                "percent": max(0, min(100, int(round(percent)))),
                "message": message if message is not None else self._progress.get("message", ""),
                "detail": detail if detail is not None else self._progress.get("detail", ""),
            }

    def get_progress(self):
        with self._progress_lock:
            return dict(self._progress)

    def deeplab_checkpoint(self, kind):
        if kind == "bw":
            return Path(self.args.bw_checkpoint or data_path("checkpoints/fakefilmbw_deeplab_best.pth"))
        return Path(self.args.color_checkpoint or data_path("checkpoints/fakefilmcolor_deeplab_best.pth"))

    def deeplab(self, kind):
        kind = "bw" if kind == "bw" else "color"
        if kind not in self._deeplab:
            self._deeplab[kind] = self.infer().load_deeplab(self.deeplab_checkpoint(kind), self.device())
        return self._deeplab[kind]

    def mask_options(self, overrides=None):
        overrides = overrides or {}

        def number(name, default, cast, min_value, max_value):
            value = overrides.get(name, default)
            try:
                value = cast(value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for {name}: {value}")
            if value < min_value or value > max_value:
                raise ValueError(f"{name} must be between {min_value} and {max_value}")
            return value

        def boolean(name, default):
            value = overrides.get(name, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.lower() in {"1", "true", "yes", "on"}:
                    return True
                if value.lower() in {"0", "false", "no", "off"}:
                    return False
            return bool(value)

        def tta_types(default):
            value = overrides.get("tta_types", default)
            if isinstance(value, str):
                selected = {item.strip() for item in value.split(",") if item.strip()}
            elif isinstance(value, dict):
                selected = {key for key, enabled in value.items() if enabled}
            else:
                selected = set(value or [])
            unsupported = selected - TTA_TYPE_CHOICES
            if unsupported:
                raise ValueError(f"Unsupported TTA types: {', '.join(sorted(unsupported))}")
            return sorted(selected)

        options = {
            "image_size": number("image_size", self.args.image_size, int, 128, 2048),
            "tile_size": number("tile_size", self.args.tile_size, int, 0, 2048),
            "tile_overlap": number("tile_overlap", self.args.tile_overlap, int, 0, 1024),
            "threshold": number("threshold", self.args.threshold, float, 0.01, 0.99),
            "close": number("close", self.args.close, int, 0, 64),
            "component_expand": number("component_expand", self.args.component_expand, int, 0, 128),
            "dilate": number("dilate", self.args.dilate, int, 0, 64),
            "mask_enhance": overrides.get("mask_enhance", self.args.mask_enhance),
            "tta": boolean("tta", self.args.tta),
            "tta_types": tta_types(self.args.tta_types),
        }
        if options["mask_enhance"] not in MASK_ENHANCE_CHOICES:
            raise ValueError(f"Unsupported mask_enhance: {options['mask_enhance']}")
        if options["tile_size"] > 0 and options["tile_overlap"] >= options["tile_size"]:
            raise ValueError("tile_overlap must be smaller than tile_size")
        return options

    def lama(self):
        if self._lama is None:
            checkpoint = Path(self.args.lama_checkpoint or data_path("model_cache/hub/checkpoints/big-lama.pt"))
            self._lama = self.infer().load_lama(checkpoint, self.device())
        return self._lama

    def make_preview(self, source_path, target_path):
        image = Image.open(source_path).convert("RGB")
        width, height = image.size
        max_preview = max(1, int(self.args.preview_max_size))
        scale = min(1.0, max_preview / max(width, height))
        if scale < 1.0:
            preview_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(preview_size, Image.Resampling.LANCZOS)
        image.save(target_path, format="JPEG", quality=90, optimize=True)
        return scale

    def refresh_result_preview(self):
        if self.result_path and self.result_path.exists():
            self.result_preview_path = self.result_path.with_name(f"{self.result_path.stem}_preview.jpg")
            self.make_preview(self.result_path, self.result_preview_path)

    def inspect_upload(self, name, data):
        suffix = Path(name or "").suffix.lower()
        try:
            with Image.open(io.BytesIO(data)) as uploaded:
                image_format = (uploaded.format or FORMAT_BY_SUFFIX.get(suffix) or "PNG").upper()
                if image_format == "MPO":
                    image_format = "JPEG"
                if image_format not in SUFFIX_BY_FORMAT:
                    image_format = FORMAT_BY_SUFFIX.get(suffix, "PNG")
                if suffix not in SUPPORTED_IMAGE_SUFFIXES or FORMAT_BY_SUFFIX.get(suffix) != image_format:
                    suffix = SUFFIX_BY_FORMAT.get(image_format, ".png")
                save_info = {}
                for key in ("icc_profile", "dpi"):
                    if key in uploaded.info:
                        save_info[key] = uploaded.info[key]
                exif = uploaded.info.get("exif")
                if exif:
                    save_info["exif"] = exif
        except Exception:
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                suffix = ".png"
            image_format = FORMAT_BY_SUFFIX.get(suffix, "PNG")
            save_info = {}
        return suffix, image_format, save_info

    def result_save_kwargs(self):
        image_format = self.source_format or FORMAT_BY_SUFFIX.get(self.result_path.suffix.lower(), "PNG")
        kwargs = {"format": image_format}
        for key, value in self.source_save_info.items():
            if key in {"icc_profile", "dpi"} or (key == "exif" and image_format in {"JPEG", "TIFF", "WEBP"}):
                kwargs[key] = value
        if image_format == "JPEG":
            kwargs.update({"quality": 100, "subsampling": 0})
        elif image_format == "TIFF":
            kwargs["compression"] = "tiff_lzw"
        elif image_format == "WEBP":
            kwargs.update({"lossless": True, "quality": 100})
        return kwargs

    def merge_result(self, base_image, result_image, mask):
        base = base_image.convert("RGB")
        result = result_image.convert("RGB")
        if result.size != base.size:
            result = result.resize(base.size, Image.Resampling.LANCZOS)
        mask_image = Image.fromarray(mask).convert("L")
        if mask_image.size != base.size:
            mask_image = mask_image.resize(base.size, Image.Resampling.NEAREST)
        merged = Image.composite(result, base, mask_image)
        source_path = self.result_path if self.result_path and self.result_path.exists() else self.image_path
        try:
            with Image.open(source_path) as source:
                if source.mode in {"RGBA", "LA"} and (self.source_format or "") in {"PNG", "TIFF", "WEBP"}:
                    alpha = source.getchannel("A")
                    if alpha.size != merged.size:
                        alpha = alpha.resize(merged.size, Image.Resampling.NEAREST)
                    merged = Image.merge("RGBA", (*merged.split(), alpha))
        except Exception:
            pass
        if (self.source_format or "") == "JPEG":
            merged = merged.convert("RGB")
        return merged

    def save_result(self, result_image, base_image, mask):
        merged = self.merge_result(base_image, result_image, mask)
        merged.save(self.result_path, **self.result_save_kwargs())

    def upload_bytes(self, name, data):
        suffix, image_format, save_info = self.inspect_upload(name, data)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.image_path = self.work_dir / f"source_{stamp}{suffix}"
        self.result_path = self.work_dir / f"result_{stamp}{suffix}"
        self.mask_path = self.work_dir / f"mask_{stamp}.png"
        self.preview_path = self.work_dir / f"preview_{stamp}.jpg"
        self.result_preview_path = None
        self.source_format = image_format
        self.source_save_info = save_info
        self.image_path.write_bytes(data)
        self.preview_scale = self.make_preview(self.image_path, self.preview_path)
        self.history = []
        self.set_progress(0, "图片已加载", active=False, detail="")

    def upload(self, name, data_url):
        self.upload_bytes(name, decode_data_url(data_url))

    def current_image(self):
        if self.result_path and self.result_path.exists():
            return Image.open(self.result_path).convert("RGB")
        if not self.image_path:
            raise ValueError("Please upload an image first")
        return Image.open(self.image_path).convert("RGB")

    def push_result_history(self):
        if self.result_path and self.result_path.exists():
            path = self.work_dir / f"history_{int(time.time() * 1000)}.png"
            Image.open(self.result_path).convert("RGB").save(path, format="PNG")
            self.history.append(path)
        else:
            self.history.append(None)

    def run_auto(self, kind):
        if not self.image_path:
            raise ValueError("Please upload an image first")
        self.set_progress(1, "准备一键除尘", detail="后台正在读取当前图像并记录撤销历史")
        self.push_result_history()
        image = self.current_image()
        infer = self.infer()
        device = self.device()
        self.set_progress(
            5,
            f"使用 {'GPU' if device.type == 'cuda' else 'CPU'} 加载除尘模型",
            detail="后台正在加载 DeepLab 灰尘分割权重",
        )
        deeplab = self.deeplab(kind)

        def mask_progress(done, total):
            total = max(total, 1)
            percent = 10 + (done / total) * 60
            self.set_progress(
                percent,
                f"正在预测灰尘区域 {done}/{total}",
                detail="后台正在逐块运行 DeepLab，并把每块概率图写回整图缓冲区",
            )

        def mask_stage(percent, message):
            self.set_progress(
                percent,
                message,
                detail="后台正在合并所有滑窗预测结果，并优化遮罩形态，方便 LaMa 精准修复",
            )

        self.set_progress(10, "开始预测灰尘区域", detail="后台正在按滑窗切分大图")
        mask = infer.predict_mask_tiled(
            deeplab,
            image,
            self.args.image_size,
            self.args.tile_size,
            self.args.tile_overlap,
            self.args.threshold,
            self.args.close,
            self.args.component_expand,
            self.args.dilate,
            self.args.mask_enhance,
            self.args.tta,
            self.args.tta_types,
            device,
            mask_progress,
            mask_stage,
        )
        self.set_progress(82, "正在保存遮罩", detail="后台正在把最终灰尘遮罩写入临时文件")
        Image.fromarray(mask).save(self.mask_path)
        self.set_progress(84, "正在加载 LaMa 修复模型", detail="后台正在准备局部图像修复模型")
        lama = self.lama()
        self.set_progress(88, "正在修复图像", detail="后台正在调用 LaMa，只重绘遮罩覆盖的灰尘和划痕区域")
        result = infer.run_lama(lama, image, mask, "crop", 800, 128, 1280)
        self.set_progress(96, "正在生成结果预览", detail="后台正在保存修复结果并生成浏览器预览图")
        self.save_result(result, image, mask)
        self.refresh_result_preview()
        self.set_progress(100, "一键除尘完成", active=False, detail="结果已生成")

    def run_auto(self, kind, options=None, mask_only=False):
        if not self.image_path:
            raise ValueError("Please upload an image first")
        self.set_progress(1, "准备生成 Mask" if mask_only else "准备一键除尘", detail="后台正在读取当前图像")
        if not mask_only:
            self.push_result_history()
        image = self.current_image()
        infer = self.infer()
        device = self.device()
        self.set_progress(
            5,
            f"使用 {'GPU' if device.type == 'cuda' else 'CPU'} 加载除尘模型",
            detail="后台正在加载 DeepLab 灰尘分割权重",
        )
        deeplab = self.deeplab(kind)
        mask_options = self.mask_options(options)

        def mask_progress(done, total):
            total = max(total, 1)
            percent = 10 + (done / total) * 60
            self.set_progress(
                percent,
                f"正在预测灰尘区域 {done}/{total}",
                detail="后台正在逐块运行 DeepLab，并把每块概率图写回整图缓冲区",
            )

        def mask_stage(percent, message):
            self.set_progress(percent, message, detail="后台正在合并滑窗结果并优化遮罩形态")

        self.set_progress(10, "开始预测灰尘区域", detail="后台正在按滑窗切分大图")
        mask = infer.predict_mask_tiled(
            deeplab,
            image,
            mask_options["image_size"],
            mask_options["tile_size"],
            mask_options["tile_overlap"],
            mask_options["threshold"],
            mask_options["close"],
            mask_options["component_expand"],
            mask_options["dilate"],
            mask_options["mask_enhance"],
            mask_options["tta"],
            mask_options["tta_types"],
            device,
            mask_progress,
            mask_stage,
        )
        self.set_progress(82, "正在保存遮罩", detail="后台正在把最终灰尘遮罩写入临时文件")
        Image.fromarray(mask).save(self.mask_path)
        if mask_only:
            self.set_progress(100, "Mask 已生成，可继续涂改", active=False, detail="Mask 已叠加到图像层上")
            return False
        self.set_progress(84, "正在加载 LaMa 修复模型", detail="后台正在准备局部图像修复模型")
        lama = self.lama()
        self.set_progress(88, "正在修复图像", detail="后台正在调用 LaMa，只重绘遮罩覆盖区域")
        result = infer.run_lama(lama, image, mask, "crop", 800, 128, 1280)
        self.set_progress(96, "正在生成结果预览", detail="后台正在保存修复结果并生成浏览器预览图")
        self.save_result(result, image, mask)
        self.refresh_result_preview()
        self.set_progress(100, "一键除尘完成", active=False, detail="结果已生成")
        return True

    def run_manual(self, mask_data_url):
        if not self.image_path:
            raise ValueError("Please upload an image first")
        self.set_progress(1, "准备手动除尘")
        self.push_result_history()
        image = self.current_image()
        mask_img = Image.open(io.BytesIO(decode_data_url(mask_data_url))).convert("RGBA")
        image_width, image_height = image.size
        if mask_img.size != (image_width, image_height):
            mask_img = mask_img.resize((image_width, image_height), Image.Resampling.NEAREST)
        arr = np.array(mask_img)
        mask = np.where((arr[:, :, 3] > 0) & (arr[:, :, 0] > 0), 255, 0).astype(np.uint8)
        if not np.any(mask):
            raise ValueError("Please paint the area to repair first")
        self.set_progress(20, "正在保存手动遮罩")
        Image.fromarray(mask).save(self.mask_path)
        device = self.device()
        self.set_progress(35, f"使用 {'GPU' if device.type == 'cuda' else 'CPU'} 加载 LaMa 修复模型")
        lama = self.lama()
        self.set_progress(70, "正在修复图像")
        result = self.infer().run_lama(lama, image, mask, "crop", 800, 128, 1280)
        self.set_progress(96, "正在生成结果预览")
        self.save_result(result, image, mask)
        self.refresh_result_preview()
        self.set_progress(100, "手动除尘完成", active=False)

    def undo_result(self):
        if not self.history:
            raise ValueError("No result to undo")
        previous = self.history.pop()
        if previous is None:
            if self.result_path and self.result_path.exists():
                self.result_path.unlink()
            self.result_preview_path = None
            return False
        img = Image.open(previous).convert("RGB")
        img.save(self.result_path, **self.result_save_kwargs())
        self.refresh_result_preview()
        return True


def send_json(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def send_png(handler, path):
    if not path or not Path(path).exists():
        handler.send_error(404)
        return
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    handler.send_response(200)
    handler.send_header("Content-Type", "image/png")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/image":
                send_png(self, app.preview_path or app.image_path)
                return
            if path == "/result":
                preview = app.result_preview_path if app.result_preview_path and app.result_preview_path.exists() else app.preview_path
                send_png(self, preview or app.image_path)
                return
            if path == "/mask":
                send_png(self, app.mask_path)
                return
            if path == "/progress":
                send_json(self, 200, app.get_progress())
                return
            if path == "/download":
                if not app.result_path or not app.result_path.exists():
                    self.send_error(404)
                    return
                data = app.result_path.read_bytes()
                suffix = app.result_path.suffix.lower()
                mime_types = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".bmp": "image/bmp",
                    ".tif": "image/tiff",
                    ".tiff": "image/tiff",
                    ".webp": "image/webp",
                }
                content_type = mime_types.get(suffix, "image/png")
                ext = suffix if suffix else ".png"
                filename = f"LAMALocal_result{ext}"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                if path == "/upload":
                    content_type = self.headers.get("Content-Type", "")
                    if content_type.startswith("multipart/form-data"):
                        form = cgi.FieldStorage(
                            fp=self.rfile,
                            headers=self.headers,
                            environ={
                                "REQUEST_METHOD": "POST",
                                "CONTENT_TYPE": content_type,
                            },
                        )
                        item = form["image"]
                        app.upload_bytes(item.filename or "image.png", item.file.read())
                    else:
                        length = int(self.headers.get("Content-Length", "0"))
                        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                        app.upload(payload.get("name", ""), payload["image"])
                    send_json(self, 200, {
                        "preview": "/image",
                        "hasResult": False,
                        "previewScale": app.preview_scale,
                        "message": "Image loaded",
                    })
                    return

                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if path == "/auto":
                    app.run_auto(payload.get("kind", "color"), payload.get("options"), mask_only=True)
                    preview_path = "/result" if app.result_path and app.result_path.exists() else "/image"
                    send_json(self, 200, {
                        "preview": preview_path,
                        "mask": "/mask",
                        "hasResult": preview_path == "/result",
                        "message": "Mask generated. You can edit it and click dust removal.",
                    })
                    return
                if path == "/manual":
                    app.run_manual(payload["mask"])
                    send_json(self, 200, {"preview": "/result", "hasResult": True, "message": "Dust removal done"})
                    return
                if path == "/undo-result":
                    has_result = app.undo_result()
                    send_json(self, 200, {
                        "preview": "/result" if has_result else "/image",
                        "hasResult": has_result,
                        "message": "Undo done",
                    })
                    return
                self.send_error(404)
            except Exception as exc:
                app.set_progress(100, f"处理失败: {exc}", active=False)
                send_json(self, 500, {"error": str(exc)})

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="LAMALocal browser dust removal app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", dest="open_browser", action="store_true", default=getattr(sys, "frozen", False))
    parser.add_argument("--no-open-browser", dest="open_browser", action="store_false")
    parser.add_argument("--work-dir", default=str(Path(os.getenv("LOCALAPPDATA", ".")) / "LAMALocal" / "web_outputs"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--color-checkpoint")
    parser.add_argument("--bw-checkpoint")
    parser.add_argument("--lama-checkpoint")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--close", type=int, default=5)
    parser.add_argument("--component-expand", type=int, default=20)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--mask-enhance", choices=sorted(MASK_ENHANCE_CHOICES), default="none")
    parser.add_argument("--tta", dest="tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--tta-types", default="flip")
    parser.add_argument("--preview-max-size", type=int, default=2400)
    return parser.parse_args()


def main():
    args = parse_args()
    app = DustApp(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(url, flush=True)
    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
