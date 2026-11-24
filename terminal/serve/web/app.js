// serve/web/app.js - the PolyStrata web page: Monitor, Requests, Images, Experts, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it
const timeStr = (t) => new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", tr("toast.copied", "Copied to clipboard"), "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "monitor";
const bootHash = location.hash;                       // the tab asked for in the address, kept for a tab that shows later (Experts)
function showTab(name) {
  tab = ["images", "monitor", "experts", "requests", "about"].includes(name) ? name : "monitor";
  if (tab === "experts" && $("tab-btn-experts").hidden) tab = "monitor";       // only when the model saves its expert profile (#5723)
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["images", "monitor", "experts", "requests", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "experts") StrataExperts.start($("view-experts")); else StrataExperts.stop();
  if (tab === "images") $("img-prompt").focus();
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "monitor" ? location.pathname : `#${tab}`);
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
  if (tab === "requests") showRequests();
  if (lastMetrics) render(lastMetrics);
}
// #142: the Requests tab opens the API monitor (/api-monitor) in a browser tab of its own, since this page's policy allows no
// frames (PAGE_CSP), or says how to turn it on. Nothing shows until /health has answered.
function showRequests() {
  $("requests-on").hidden = health.api_monitor !== true;
  $("requests-off").hidden = health.api_monitor !== false;
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
let apiKey = store.get("apikey", "");
function headers(json = false) {
  const h = {};
  if (apiKey) h.Authorization = "Bearer " + apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = apiKey;
function saveApiKey(notify = false) {
  apiKey = $("api-key").value.trim();                // usable even when browser storage is blocked or full
  const saved = store.set("apikey", apiKey);
  if (notify) {
    if (saved) toast("success", "API key saved", "Kept in this browser only.");
    else toast("warn", "API key used for this page only", "Browser storage is unavailable. Enter it again after reloading.");
  }
}
$("api-key").oninput = () => saveApiKey();
$("api-key").onchange = () => saveApiKey(true);

let health = {model: "strata", images: false, max_context: 0};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    $("tab-btn-images").hidden = !health.image_generation;
    if (location.hash === "#images" && health.image_generation) showTab("images");
    if (tab === "requests") showRequests();
    StrataExperts.init({headers, timeStr});
    StrataExperts.probe().then((on) => {
      $("tab-btn-experts").hidden = !on;
      if (on && bootHash === "#experts" && tab === "monitor") showTab("experts");
      if (!on && tab === "experts") showTab("monitor");                // the model list switched to a model without a profile
      fitHeader();
    });
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ the model list (GET api/models, POST api/models/load)
// The installed models, shown once there is more than one to choose from. Choosing one stops the model in memory and loads
// that one.
async function loadModels() {
  const sel = $("model-select");
  let models = [];
  try {
    const r = await fetch("api/models", {headers: headers()});
    if (r.ok) models = (await r.json()).models || [];
  } catch (e) { return; }
  const unavailable = tr("model.unavailable", "unavailable");
  sel.innerHTML = models.map((m) =>
    `<option value="${esc(m.id)}"${m.current ? " selected" : ""}${m.status === "unavailable" ? ` disabled title="${esc(m.problem || "")}"` : ""}>` +
    `${esc(m.display_name && m.display_name !== m.id ? `${m.display_name} (${m.id})` : m.id)}` +
    `${m.status === "unavailable" ? ` - ${esc(unavailable)}` : ""}</option>`).join("");
  sel.hidden = models.length < 2;
  sel.dataset.current = (models.find((m) => m.current) || {}).id || "";
  fitHeader();
}
$("model-select").onchange = async () => {
  const sel = $("model-select"), name = sel.value;
  sel.disabled = true;
  toast("info", tr("model.switching", "Switching to {name}", {name}),
        tr("model.switchingText", "The model in memory is stopped first, then this one is loaded."), 8000);
  let message = "";
  try {
    const r = await fetch("api/models/load", {method: "POST", headers: headers(true), body: JSON.stringify({model: name})});
    if (!r.ok) message = ((await r.json().catch(() => ({}))).error || {}).message || tr("toast.httpAnswer", "The server answered HTTP {code}.", {code: r.status});
  } catch (e) { message = tr("model.noAnswer", "The server did not answer."); }
  sel.disabled = false;
  await loadHealth();
  await loadModels();
  if (message) { toast("error", tr("model.notSwitched", "The model was not switched"), message, 10000); return; }
  toast("success", tr("model.switched", "{name} is loaded", {name}), "");
};

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "model", label: "Model", icon: "info", unit: ""},
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  // #5484: what the load is (NVML, amdgpu and the Mac report time busy, not compute in use)
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100,
   hint: "Share of the time the GPU had work to do. It does not say how much of its compute is in use."},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card${m.key === "model" ? " model-card" : ""}"><div class="st-metric">
    <span class="st-metric__label" id="ml-${m.key}"${m.hint ? ` title="${esc(m.hint)}"` : ""}>${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span><span class="st-metric__sub speed-range" id="sr-speed"></span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span><span class="st-metric__sub speed-range" id="sr-prefill"></span></div>
    </div>` : m.key === "model" ? `<span class="st-metric__value model-card__name" id="mv-model">–</span>
    <span class="st-metric__sub" id="ms-model"></span>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>${m.key === "temp" || m.key === "power" ? `<span class="st-metric__sub speed-range" id="sr-${m.key}"></span>` : ""}`}
    ${m.key === "model" ? `<div class="model-card__facts" id="model-facts"></div>` : `<svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
      <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
      stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>`}</div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub, why) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
  // why a card has no reading (the server's gpu_note), on its line and as its tooltip (#5457)
  $(`ms-${key}`).title = why || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let metricsEpoch = 0;
// After Load or Unload returns, a poll that started earlier can still say the opposite. Hold the POST's state until a later
// poll agrees, or until the first poll that started after the POST arrives. (PR #3550 by sakuumomo)
let metricsHold = null;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
function metricsAgree(live, state) {
  const s = (live && live.state) || "";
  if (state === "unloaded") return s === "unloaded";
  return s !== "" && s !== "unloaded";
}
async function poll() {
  const epoch = ++metricsEpoch;
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", tr("pill.keyNeeded", "API key needed"));
      if (!keyWarned) { keyWarned = true; toast("warn", tr("pill.keyNeeded", "API key needed"), tr("toast.keyAbout", "This server needs a key: add it under About > Settings."), 6000); }
    } else if (r.ok) {
      const m = await r.json();
      if (metricsHold && epoch <= metricsHold.after && !metricsAgree(m.live, metricsHold.state)) {
        setTimeout(poll, 1000);
        return;
      }
      if (metricsHold && (metricsAgree(m.live, metricsHold.state) || epoch > metricsHold.after)) metricsHold = null;
      lastMetrics = m;
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", tr("pill.unreachable", "Server not reachable"));
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
  // a poll re-sets the sentence every second; refit only when its text changed
  if (text !== lastPillText) { lastPillText = text; fitHeader(); }
}
let lastPillText = null;
// The pill's sentence is the first thing the header row can spare when it runs out of room.
function fitHeader() {
  const header = document.querySelector(".st-header");
  if (!header) return;
  header.classList.remove("st-header--clip");
  if (header.scrollWidth > header.clientWidth) header.classList.add("st-header--clip");
}
window.addEventListener("resize", fitHeader);

// ------------------------------------------------------------------ Load / Unload (POST /load, /unload: the header button)
// Metrics stay "unloaded" until the engine is READY, so an image job on this page has to paint Loading itself.
// modelOp covers the button's own POST, which returns before the next poll can agree.
let modelOp = null;
let imageBusy = false;
const pageBusy = () => imageBusy;
// "generating an image" is the job a refresh can see. imageBusy is this page's POST before that phase is published. A text
// request already on the metrics wins.
function imageLive(live) {
  if (live && live.phase === "generating an image") return true;
  const state = live && live.state;
  if (state === "reading" || state === "generating") return false;
  return imageBusy;
}
function paintLive() { if (lastMetrics) render(lastMetrics); }
function turnBlocksUnload(live) {
  const state = (live && live.state) || "";
  return pageBusy() || state === "reading" || state === "generating" || (live && live.queued > 0);
}
function setModelButton(text, title, disabled) {
  const btn = $("unload-btn");
  btn.textContent = text; btn.title = title; btn.disabled = disabled;
  fitHeader();
}
function modelButton(live) {
  if (modelOp === "load") return setModelButton(tr("model.loading", "Loading…"), tr("model.loadingTitle", "The model is loading"), true);
  if (modelOp === "unload") return setModelButton(tr("model.unloading", "Unloading…"), tr("model.unloadingTitle", "The model is unloading"), true);
  const state = (live && live.state) || "";
  const metricsBusy = state === "reading" || state === "generating" || (live && live.queued > 0);
  const imageNow = imageLive(live);
  const unload = tr("model.unload", "Unload");
  if (state === "unloaded") {
    if (imageNow) return setModelButton(tr("model.loading", "Loading…"), tr("model.imageLoading", "An image is already loading the model"), true);
    return setModelButton(tr("model.load", "Load"), tr("model.loadTitle", "Load the model now. A message also loads it."), false);
  }
  if (imageNow) return setModelButton(unload, tr("model.imageBusy", "An image is generating. It finishes on its own."), true);
  if (!state) return setModelButton("…", tr("model.waiting", "Waiting for the server"), true);
  if (metricsBusy || pageBusy()) return setModelButton(unload, tr("model.stopFirst", "Stop the reply, then unload"), true);
  return setModelButton(unload, tr("model.unloadTitle", "Unload the model and free the GPU"), false);
}
$("unload-btn").onclick = async () => {
  if (modelOp) return;
  const live = (lastMetrics && lastMetrics.live) || {};
  const loading = live.state === "unloaded";
  const busyCopy = (sample, stillLoading) => stillLoading
    ? (imageLive(sample) ? tr("model.imageLoadingDot", "An image is already loading the model.") : tr("model.replyLoadingDot", "A reply is already loading the model."))
    : (imageLive(sample) ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
  if (loading ? pageBusy() : turnBlocksUnload(live)) {
    toast("warn", tr("model.busy", "Model is busy"), busyCopy(live, loading));
    return;
  }
  if (!loading && !confirm(tr("model.confirmUnload", "Unload the model and free the GPU? Load, or the next message, brings it back."))) return;
  // confirm() blocks the poll. Yield so a sample that arrived during the dialog is applied. A confirmed Unload stays an unload.
  if (!loading) {
    modelOp = "unload";
    modelButton(live);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const now = (lastMetrics && lastMetrics.live) || {};
  const loadingNow = loading;
  if (loadingNow ? pageBusy() : turnBlocksUnload(now)) {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton(now);
    toast("warn", tr("model.busy", "Model is busy"), busyCopy(now, loadingNow));
    return;
  }
  modelOp = loadingNow ? "load" : "unload";
  modelButton(now);
  if (loadingNow) toast("info", tr("model.loadingToast", "Loading the model"), tr("model.fewMinutes", "This can take a few minutes."));
  const remember = (state) => {
    if (!lastMetrics) lastMetrics = {};
    lastMetrics.live = Object.assign({}, lastMetrics.live, {state, queued: 0});
    metricsHold = {state, after: metricsEpoch};
  };
  try {
    const r = await fetch(loadingNow ? "load" : "unload", {method: "POST", headers: headers(true), body: "{}"});
    let body = {};
    try { body = await r.json(); } catch (e) { /* the connection dropped */ }
    const status = body.status || "";
    const msg = (body.error && body.error.message) ||
                (r.status === 401 ? tr("toast.keyAbout2", "API key needed: add it under About > Settings.") : tr("toast.httpAnswer", "The server answered HTTP {code}.", {code: r.status}));
    const failTitle = loadingNow ? tr("model.loadFailed", "Load failed") : tr("model.unloadFailed", "Unload failed");
    if (loadingNow) {
      if (r.ok && status === "loaded") {
        remember("idle");
        toast("success", tr("model.loaded", "Model loaded"), tr("model.gpuInUse", "The GPU is in use."));
      } else {
        toast("error", failTitle, msg);
      }
    } else if (status === "unloaded" || status === "not loaded") {
      remember("unloaded");
      toast(status === "unloaded" ? "success" : "info",
            status === "unloaded" ? tr("model.unloaded", "Model unloaded") : tr("model.alreadyUnloaded", "Already unloaded"),
            tr("model.gpuFree", "The GPU is free. Load, or the next message, brings it back."));
    } else if (r.status === 409 || status === "busy") {
      let sample = (lastMetrics && lastMetrics.live) || {};
      try {
        const mr = await fetch("metrics", {headers: headers()});
        if (mr.ok) {
          const m = await mr.json();
          if (m && m.live) { sample = m.live; lastMetrics = m; }
        }
      } catch (e) { /* the 409 body has no phase */ }
      toast("warn", tr("model.busy", "Model is busy"), imageLive(sample)
        ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
    } else if (status === "unsupported") {
      toast("error", tr("model.unsupported", "Unload is not available"), "");
    } else {
      toast("error", failTitle, msg);
    }
  } catch (e) {
    toast("error", loadingNow ? tr("model.loadFailed", "Load failed") : tr("model.unloadFailed", "Unload failed"), tr("model.noAnswer", "The server did not answer."));
  } finally {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton({});
  }
};

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill. modelOp and a local load win over "N queued": with parallel requests, queued stays up through
  // ensure_loaded(), and metrics stay "unloaded" until the engine is READY. An image POST in that window is still a load.
  const localLoad = live.state === "unloaded" && imageBusy;
  if (modelOp === "load") {
    setPill("idle", tr("model.loading", "Loading…"));
  } else if (modelOp === "unload") {
    setPill("idle", tr("model.unloading", "Unloading…"));
  } else if (localLoad) {
    setPill("idle", tr("model.loading", "Loading…"));
  } else if (imageLive(live)) {
    setPill("reading", tr("pill.image", "Generating image…"));
  } else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? tr("pill.readingPct", "Reading prompt · {pct}%", {pct}) : tr("pill.reading", "Reading prompt"));
  } else if (live.state === "generating") {
    setPill("generating", tr("pill.generating", "Generating · {rate} tok/s", {rate: fmt(live.tok_s, 1)}));
  } else if (live.state === "unloaded") {
    setPill("idle", tr("pill.unloaded", "Unloaded"));
  } else {
    setPill("idle", tr("pill.idle", "Idle"));
  }
  modelButton(live);
  if (!modelOp && !localLoad && live.queued > 0) setPill("queued", tr("pill.queued", "{n} queued", {n: live.queued}));
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
/* The machine's PCIe link for the PCIe card: with several cards the one with the slowest link (gpu_pcie_bottleneck, which the server names
   from ONE card: its generation, width and the generation it can reach), else the top-level fields (one card, or a server from before it).
   gen is the generation the link can reach (NVML reports a lower one while a card is idle), cur the one it runs at now. */
function pcieLink(hw) {
  const b = hw && hw.gpu_pcie_bottleneck;
  if (b && (b.gen || b.gen_max) && b.width) return { gen: b.gen_max || b.gen, cur: b.gen || null, width: b.width, card: b.card };
  if (!hw) return { gen: null, cur: null, width: null, card: null };
  return { gen: hw.gpu_pcie_gen_max || hw.gpu_pcie_gen || null, cur: hw.gpu_pcie_gen || null, width: hw.gpu_pcie_width || null, card: null };
}

// The Speed card's min and max (#4968, from PR #4968 by noahark): the slowest and the fastest of the newest SPEED_KEPT requests, the same
// window the server keeps (GET /metrics?requests=all, newest last in its deque of 500). The page adds every polled request to its own
// map and drops the oldest past SPEED_KEPT, so an extreme the server has let go of is gone here too. A new totals window (a server
// restart) starts the map afresh from one full pull.
const SPEED_KEPT = 500;
let speedWin = new Map(), speedWinSince = null;
function keepNewest(map, cap) {
  if (map.size <= cap) return map;
  const rows = [...map.values()].sort((a, b) => (a.time || 0) - (b.time || 0));
  return new Map(rows.slice(-cap).map((r) => [r.id, r]));
}
const prefillOf = (r) => (r.prompt_ms > 0 && r.prompt_tokens != null && r.prompt_tokens - (r.reused || 0) > 0
                          ? (r.prompt_tokens - (r.reused || 0)) / (r.prompt_ms / 1000) : null);
function speedRange(rows) {
  const out = {};
  for (const [key, get] of [["decode", (r) => r.decode_tok_s], ["prefill", prefillOf]]) {
    const v = rows.map(get).filter((x) => x != null && x > 0);
    out[key] = v.length ? {min: Math.min(...v), max: Math.max(...v), n: v.length} : null;
  }
  return out;
}
async function pullSpeedWindow() {
  try {
    const r = await fetch("metrics?requests=all", {headers: headers()});
    if (r.ok) for (const q of ((await r.json()).requests || [])) if (q.id != null) speedWin.set(q.id, q);
    speedWin = keepNewest(speedWin, SPEED_KEPT);
    fillSpeedRange();
  } catch { /* the window stays as it is; the next polls add to it */ }
}
function fillSpeedRange() {
  const range = speedRange([...speedWin.values()]);
  for (const [id, key, d] of [["sr-speed", "decode", 1], ["sr-prefill", "prefill", 0]]) {
    const r = range[key], el = $(id);
    el.textContent = r ? `min ${fmt(r.min, d)} · max ${fmt(r.max, d)}` : "";
    el.title = r ? `The slowest and the fastest of the last ${fmt(r.n)} requests. The server keeps up to 500.` : "";
  }
}
// Power and temperature averages (#4968): a sample is taken at each poll while the server is working (queued, reading or generating;
// idle time is left out) and weighs the seconds since the poll before it, at most HW_GAP_S, so a page that polled less often (a
// background tab, another tab of this page) neither drops that time nor lets one sample stand for minutes. A new totals window or
// another model starts the averages afresh. They cover what this page has watched, and their tooltip says how long that was.
const HW_GAP_S = 2;
let hwWin = [], hwKey = null, hwLast = null;
function hwAverages(samples) {
  let secs = 0, pw = 0, ps = 0;
  const cols = samples.reduce((n, s) => Math.max(n, s.t.length), 0), tw = Array(cols).fill(0), ts = Array(cols).fill(0);
  for (const s of samples) {
    secs += s.w;
    if (s.p != null) { pw += s.w; ps += s.w * s.p; }
    s.t.forEach((x, c) => { if (x != null) { tw[c] += s.w; ts[c] += s.w * x; } });
  }
  return {seconds: secs, power: pw ? ps / pw : null, temps: tw.map((w, c) => (w ? ts[c] / w : null))};
}
function hwSample(win, last, now, hw, working) {
  const w = last == null ? 1 : Math.min(Math.max(now - last, 0), HW_GAP_S);
  if (!working || !w) return win;
  const g = [...(hw.gpus || [])].sort((a, b) => (a.index || 0) - (b.index || 0));
  if (!g.length && hw.gpu_power == null) return win;
  win.push({w, t: g.map((x) => x.temp), p: hw.gpu_power, names: g.map((x) => x.name)});
  return win.length > 21600 ? win.slice(-18000) : win;
}
const spanText = (s) => (s < 90 ? `${fmt(s)} s` : s < 5400 ? `${fmt(s / 60)} min` : `${fmt(s / 3600, 1)} h`);
function sampleHw(hw, working, key) {
  const now = Date.now() / 1000;
  if (hwKey !== key) { hwWin = []; hwKey = key; hwLast = null; }
  hwWin = hwSample(hwWin, hwLast, now, hw, working);
  hwLast = now;
  const avg = hwAverages(hwWin), span = spanText(avg.seconds);
  const pe = $("sr-power");
  if (pe) {
    pe.textContent = avg.power != null ? `avg ${fmt(avg.power)} W` : "";
    pe.title = avg.power != null ? `Average power while the server was working, over the ${span} this page has watched. Idle time is left out.` : "";
  }
  const te = $("sr-temp");
  if (te) {
    const any = avg.temps.some((x) => x != null);
    te.textContent = any ? `avg ${avg.temps.map((x) => (x == null ? "–" : fmt(x))).join(" / ")}` : "";
    const names = hwWin.length ? hwWin[hwWin.length - 1].names : [];
    const detail = avg.temps.map((x, i) => (x == null || !names[i] ? null : `${names[i]}: ${fmt(x)} °C`)).filter(Boolean).join(" · ");
    te.title = any ? `${detail ? detail + "\n" : ""}Average temperature of each card while the server was working, over the ${span} this page has watched. Idle time is left out.` : "";
  }
}

// The Context fill gauge (#4636, from PR #4636 by thisisspartan): the window split into what the engine already had cached, what is
// being read, the answer so far and what is free. While a request runs it reads the live row (reused is the RESUME prefix; prompt_read
// counts the reused part too); idle, the last request's prompt and answer show as cached, but only while that request's engine
// process is the one running now (after a restart the cache is gone). A request larger than the window scales the parts down and turns
// the percent red; 85 % and above is amber. `on` is the Model-state key: reading and generating light their row, a queued request says so.
function contextParts(live, eng, last, on) {
  const ctx = eng.max_context || 0;
  let cached = 0, read = 0, answer = 0;
  if (live.state === "reading" || live.state === "generating") {
    const p = live.prompt_tokens || 0;
    cached = Math.min(live.reused || 0, p);
    read = live.state === "generating" ? p - cached
         : Math.min(Math.max(0, (live.prompt_read || 0) - cached), p - cached);
    answer = live.state === "generating" ? Math.max(0, live.generated || 0) : 0;
  } else if (live.state === "idle" && last && !(eng.gen != null && last.gen != null && last.gen !== eng.gen)) {
    cached = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  }
  const used = cached + read + answer, scale = ctx > 0 && used > ctx ? ctx / used : 1;
  const pct = ctx ? Math.round(Math.min(1, used / ctx) * 100) : null;
  return {ctx, cached, read, answer, used, free: Math.max(0, ctx - used), scale, pct,
          tone: ctx > 0 && used > ctx ? "danger" : pct != null && pct >= 85 ? "warn" : "",
          now: on === "reading" ? "read" : on === "generating" ? "answer" : null, queued: on === "queued"};
}
function renderContext(live, eng, last, on) {
  const c = contextParts(live, eng, last, on), ARC = 235.6;
  let at = 0;
  for (const k of ["cached", "read", "answer"]) {
    const len = c.ctx ? ARC * c[k] * c.scale / c.ctx : 0, el = $(`ctx-seg-${k}`);
    el.setAttribute("stroke-dasharray", `${len.toFixed(1)} 314.2`);
    el.setAttribute("stroke-dashoffset", `${(-at).toFixed(1)}`);
    el.style.opacity = len >= 1 ? "1" : "0";
    at += len;
  }
  $("ctx-pct").textContent = c.pct == null ? "–" : `${c.pct}%`;
  $("ctx-pct").dataset.tone = c.tone;
  $("ctx-sub").textContent = c.ctx ? `${kfmt(c.used)} / ${ctxfmt(c.ctx)}` : "–";
  for (const k of ["cached", "read", "answer", "free"]) {
    $(`ctx-leg-${k}`).textContent = c.ctx || k !== "free" ? kfmt(c[k]) : "–";
    $(`ctx-row-${k}`).classList.toggle("is-now", c.now === k);
  }
  $("ctx-note").hidden = !c.queued;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const imageNow = imageLive(live);
  const starting = modelOp === "load" || imageNow;
  const on = !modelOp && live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) {
    const lit = (b.dataset.s === on || b.dataset.s === live.state) && !(starting && b.dataset.s === "unloaded");
    b.classList.toggle("on", lit);
  }
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (modelOp === "load") {
    label = "Loading";
    delete prog.dataset.tone;
    detail = "The model is loading.";
  } else if (imageNow && (live.state === "unloaded" || live.state === "reading" || live.state === "generating" || live.state === "idle")) {
    label = live.state === "unloaded" ? "Loading" : "Generating image";
    delete prog.dataset.tone;
    detail = live.state === "unloaded" ? "The model is loading." : "An image is generating. It finishes on its own.";
  } else if (modelOp === "unload") {
    label = "Unloading";
    delete prog.dataset.tone;
    detail = "Freeing the GPU.";
  } else if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (live.state === "unloaded") {
    label = "Unloaded";
    delete prog.dataset.tone;
    detail = "Load brings it back. A message does too.";
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  // Idle: not the last request's speed (one request is noisy) but the run's average from the server's totals
  // (token-weighted, the same figure the engine's DONE lines and Prometheus report). Live stays live.
  const avgDecode = totals && totals.decode_ms > 0 && totals.output_tokens > 0 ? totals.output_tokens / (totals.decode_ms / 1000) : null;
  const readTokens = totals ? totals.prompt_tokens - (totals.reused || 0) : 0;
  const avgPrefill = totals && totals.prompt_ms > 0 && readTokens > 0 ? readTokens / (totals.prompt_ms / 1000) : null;
  const speed = live.state === "generating" ? live.tok_s : avgDecode != null ? avgDecode : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : avgDecode != null ? "Decode · run average" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : avgPrefill != null ? avgPrefill : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : avgPrefill != null ? "Prefill · run average" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  if (speedWinSince !== (totals && totals.since)) { speedWin = new Map(); speedWinSince = totals && totals.since; pullSpeedWindow(); }
  for (const r of requests) if (r.id != null) speedWin.set(r.id, r);
  speedWin = keepNewest(speedWin, SPEED_KEPT);
  fillSpeedRange();
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : (st.gpu_note ? "not available on Windows AMD yet" : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  // #5457: a card the server cannot read (a Mac has no temperature or power without root and no PCIe link; a Windows AMD card)
  // says so, with the server's own reason, whether or not the GPU has a name
  const unread = (v) => v == null && !multi && !!st.gpu_note;
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : unread(hw.gpu_temp) ? "not available" : "",
            unread(hw.gpu_temp) ? st.gpu_note : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W",
            multi ? per((g) => (g.power == null ? "–" : `${fmt(g.power)} W` +
                                             (g.power_limit ? ` of ${fmt(g.power_limit)} W limit` : "")))
                  : hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : unread(hw.gpu_power) ? "not available" : "",
            unread(hw.gpu_power) ? st.gpu_note : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  // each card trains its own link: with several cards the big number is the weakest card's and the line under it
  // names every card, because one card's "Gen5 x8" said nothing about a second card trained down to x4
  // (the per-card words follow the big number's rule: the link the card trained, not the lower one it idles at)
  // a card that runs narrower than it could - capped to x8 in a x16 slot, or trained down by the board - says both,
  // because the narrow number alone reads as a link someone chose
  const link = (g) => { const v = g.pcie_gen_max || g.pcie_gen;
                        return v ? `Gen${v}${g.pcie_width ? ` x${g.pcie_width}` +
                               (g.pcie_width_max > g.pcie_width ? ` of x${g.pcie_width_max}` : "") : ""}` : "–"; };
  // the big number with several cards: the one card whose link carries the least (gpu_pcie_bottleneck); the top-level fields are card 0's
  const slow = multi && hw.gpu_pcie_bottleneck ? hw.gpu_pcie_bottleneck : null;
  const one = slow ? { gen: slow.gen, gen_max: slow.gen_max, width: slow.width, width_max: slow.width_max }
                   : { gen: hw.gpu_pcie_gen, gen_max: hw.gpu_pcie_gen_max, width: hw.gpu_pcie_width, width_max: hw.gpu_pcie_width_max };
  const gen = one.gen_max || one.gen;
  const width = one.width ? `x${one.width}` + (one.width_max > one.width ? ` of x${one.width_max}` : "") : "";
  const rx = hw.gpu_pcie_rx_mb == null ? ""
           : `${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s`;
  setMetric("pcie", gen ? `Gen${gen}` : null, width,
            multi ? `${per(link)}${rx ? ` · ${rx} total` : ""}`
                  : rx === "" ? (unread(gen || null) ? "not available" : "") : `to GPU ${rx}` +
                  (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""),
            unread(gen || null) ? st.gpu_note : "");
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);
  // power and temperature averages (#4968); after the cards block above so that block stays self-contained
  sampleHw(hw, !!live.state && live.state !== "idle" && live.state !== "unloaded", `${totals && totals.since}|${eng.model || ""}`);

  // the model card (#4968, from PR #4968 by noahark): which model this is (the name appears nowhere else on this tab) and the averages of
  // the server's own totals since it started. Tooltips are plain title= text, like the rest of this tab.
  $("ml-model").title = "Which model this server runs. Apps ask for it by this exact name.";
  $("mv-model").textContent = eng.model || "–";
  $("mv-model").title = eng.model || "";
  $("ms-model").textContent = [eng.max_context ? `${ctxfmt(eng.max_context)} context` : null,
                               eng.version ? `engine v${eng.version}` : null].filter(Boolean).join(" · ");
  $("ms-model").title = "The context window this model runs with, and the engine's version.";
  const avg = [];
  if (totals && totals.requests) {
    const read = totals.prompt_tokens - totals.reused;
    avg.push(["Decode avg", totals.decode_ms > 0 && totals.output_tokens ? `${fmt(totals.output_tokens / (totals.decode_ms / 1000), 1)} t/s` : "–",
              "Average answer speed since the server started: answer tokens divided by the time spent writing them. Waiting in the queue and reading the prompt are not counted."]);
    avg.push(["Prefill avg", totals.prompt_ms > 0 && read > 0 ? `${fmt(read / (totals.prompt_ms / 1000))} t/s` : "–",
              "Average prompt speed since the server started, counting only the tokens that were really read. What came from the cache is left out."]);
    avg.push(["Per request", `${fmt(totals.prompt_tokens / totals.requests)} in · ${fmt(totals.output_tokens / totals.requests, 1)} out`,
              "The average prompt and answer length of one request."]);
    avg.push(["Prompt reused", totals.prompt_tokens ? `${fmt((100 * totals.reused) / totals.prompt_tokens)}%` : "–",
              "How much of each prompt came from the engine's cache instead of being read again. High means follow-ups are fast."]);
    const t0 = new Date(totals.since * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
    avg.push(["Since", `${t0} · ${fmt(totals.requests)} requests`,
              "When this server began counting, and how many requests it has served. Loading or unloading the model does not reset it."]);
  }
  $("model-facts").innerHTML = avg.length
    ? avg.map(([k, v, tip]) => `<span class="mf-k" title="${esc(tip)}">${esc(k)}</span><span class="mf-v">${esc(v)}</span>`).join("")
    : `<span class="mf-k mf-k--wide">Averages start with the first request.</span>`;

  // context fill (#4636): the running request in its parts, else the last one
  renderContext(live, eng, last, on);
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  // #5484: the VRAM hit rate column only when a request has one (a Mac and a model wholly in VRAM have no expert cache to hit)
  const anyHit = requests.some((r) => r.hit_rate != null);
  $("req-hit").hidden = !anyHit;
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="${anyHit ? 9 : 8}" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"],
                   timeout: ["st-badge--queued", "Timeout"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        ${anyHit ? `<td class="num">${hit}</td>` : ""}<td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    // #5080: the MTP line only when the MTP is the drafter (eng.drafting, strata_control/drafting.py engine_info; an engine too old to say had only the MTP)
    ["Speculation", eng.spec && (eng.drafting == null || eng.drafting === "mtp") ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    // #5457: the server's reason for the readings it lacks is shown beside a named GPU too (a Mac always has a name)
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}${st.gpu_note ? ` · ${st.gpu_note}` : ""}` : (st.gpu_note || "not readable (NVML)")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config, offered to the model on requests that ask for them ("strata_mcp": true);
// the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ language (serve/web/i18n.js)
// The user's own choice, kept in this browser: English until they pick another. /?lang=de shows one for this visit only.
const urlLang = new URLSearchParams(location.search).get("lang");
$("lang").innerHTML = StrataI18n.LANGUAGES.map(([code, name]) => `<option value="${esc(code)}">${esc(name)}</option>`).join("");
function showLanguage() {
  $("lang").value = StrataI18n.lang;
  if (lastMetrics) render(lastMetrics);
  loadModels();                                         // "unavailable" in the model list
}
$("lang").onchange = async () => {
  store.set("lang", $("lang").value);
  await StrataI18n.setLanguage($("lang").value);
  showLanguage();
};

// ------------------------------------------------------------------ start
async function start() {
  await StrataI18n.setLanguage(urlLang || store.get("lang", "en"));
  $("lang").value = StrataI18n.lang;
  loadHealth().then(loadMcp);
  loadModels();
  showTab(location.hash.slice(1) || "monitor");
  poll();
  requestAnimationFrame(fitHeader);
}
start();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json"};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  imageBusy = true;
  $("img-go").disabled = true;
  paintLive();
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    imageBusy = false;
    $("img-go").disabled = false;
    paintLive();
  }
};
