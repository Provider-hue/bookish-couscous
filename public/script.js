/* rndf frontend: fetches the backend URL + key from this deployment's env
 * (/api/config -> BACKEND_URL, DEMO_API_KEY), lets the user override them, then
 * drives the sahu65 backend (Render): GET /readyz for status, POST /detect for
 * predictions. No frameworks - vanilla JS only. */
"use strict";

const $ = (id) => document.getElementById(id);
const LS_URL = "rndf.backendUrl";
const LS_KEY = "rndf.apiKey";

const config = { backendUrl: "", apiKey: "" };

// Meter values waiting on the iframe to finish loading.
let pendingMeter = null;

// --- config -----------------------------------------------------------------

async function fetchEnvConfig() {
  try {
    const res = await fetch("/api/config", { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      config.backendUrl = (data.backendUrl || "").trim();
      config.apiKey = (data.apiKey || "").trim();
    }
  } catch (e) {
    // No serverless env available (e.g. opened as a plain file) - manual input still works.
  }
}

function applyConfigToInputs() {
  $("backend-url").value = localStorage.getItem(LS_URL) || config.backendUrl || "";
  $("api-key").value = localStorage.getItem(LS_KEY) || config.apiKey || "";
}

function readInputsIntoConfig() {
  config.backendUrl = ($("backend-url").value || config.backendUrl || "").trim().replace(/\/+$/, "");
  config.apiKey = ($("api-key").value || config.apiKey || "").trim();
}

function saveConfig() {
  readInputsIntoConfig();
  if (config.backendUrl) localStorage.setItem(LS_URL, config.backendUrl);
  if (config.apiKey) localStorage.setItem(LS_KEY, config.apiKey);
  checkBackend();
}

function authHeaders() {
  const h = {};
  if (config.apiKey) {
    h["Authorization"] = `Bearer ${config.apiKey}`;
    h["X-API-Key"] = config.apiKey;
  }
  return h;
}

// --- backend status ---------------------------------------------------------

function setStatus(kind, text) {
  const el = $("status");
  el.className = `badge badge-${kind}`;
  el.textContent = text;
}

async function checkBackend() {
  readInputsIntoConfig();
  if (!config.backendUrl) {
    setStatus("unknown", "no backend URL");
    return;
  }
  try {
    const res = await fetch(`${config.backendUrl}/readyz`, { headers: authHeaders() });
    if (res.status === 200) {
      setStatus("ready", "backend ready");
    } else if (res.status === 401) {
      setStatus("error", "401 - bad API key");
    } else if (res.status === 503) {
      setStatus("loading", "model loading");
    } else {
      setStatus("error", `backend ${res.status}`);
    }
  } catch (e) {
    setStatus("error", "backend unreachable");
  }
}

// --- detection --------------------------------------------------------------

function showError(message) {
  const el = $("error");
  if (!message) {
    el.classList.add("hidden");
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

// --- result rendering ---------------------------------------------------------

function verdictClass(verdict) {
  if (verdict === "likely_ai") return "verdict-ai";
  if (verdict === "ai_generated_verified") return "verdict-verified";
  if (verdict === "likely_real") return "verdict-real";
  return "verdict-inconclusive";
}

// The response reports `summary.verdict` in plain language (MANIPULATED / AUTHENTIC /
// INCONCLUSIVE / AI_GENERATED_VERIFIED) and carries the detector's own label
// alongside it as `summary.raw_verdict`. The chips on screen keep showing the
// detector's label, so the two never disagree about what was actually measured.
function setMeter(percent, caption) {
  const frame = $("meter");
  let meter = null;
  try {
    meter = frame.contentWindow && frame.contentWindow.rndfMeter;
  } catch (e) {
    // Different origin: the meter still accepts postMessage.
  }
  if (meter) {
    meter.setScore(percent);
    if (caption) meter.setCaption(caption);
    pendingMeter = null;
    return;
  }

  // The frame has not finished loading yet. Hold the values and apply them the
  // moment it reports ready, so the first scan never renders a meter stuck at 0.
  pendingMeter = { score: percent, caption };
  try {
    frame.contentWindow.postMessage({ type: "rndf:meter-score", score: percent }, "*");
    if (caption) {
      frame.contentWindow.postMessage({ type: "rndf:meter-caption", text: caption }, "*");
    }
  } catch (e) {
    // Frame not reachable at all; the next successful scan will retry.
  }
}

function flushPendingMeter() {
  if (!pendingMeter) return;
  setMeter(pendingMeter.score, pendingMeter.caption);
}

function renderSignals(signals) {
  const el = $("signals");
  el.textContent = "";
  if (!Array.isArray(signals) || signals.length === 0) {
    el.textContent = "-";
    return;
  }
  for (const name of signals) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = name;
    el.appendChild(chip);
  }
}

function renderNarrative(explanation, intent, provider) {
  $("explanation").textContent = explanation || "-";
  $("intent").textContent = intent || "No intent inference was produced for this scan.";

  const note = $("narration-note");
  if (provider) {
    note.textContent = `Explanation and intent narrated by ${provider}.`;
    note.classList.remove("hidden");
  } else {
    note.textContent =
      "Narrator unavailable - explanation is a factual restatement of the detector output, " +
      "and intent was not inferred.";
    note.classList.remove("hidden");
  }
}

function renderResult(data) {
  const summary = data.summary || {};
  const visual = (data.ensemble_results || {}).visual_manipulation || {};
  const meta = data.metadata_analysis || {};

  const rawVerdict = summary.raw_verdict || "inconclusive";
  const score = Number(summary.overall_score ?? 0);

  const v = $("verdict");
  v.textContent = rawVerdict;
  v.className = `verdict ${verdictClass(rawVerdict)}`;

  setMeter(Math.round(score * 100), "0 = authentic · 100 = AI-generated");

  // sahu65 1.1.0 renamed `classifier_probability` to `detector_probability`; the
  // label on screen is unchanged.
  const detector = visual.detector_probability;
  $("classifier").textContent =
    detector !== null && detector !== undefined ? Number(detector).toFixed(4) : "-";
  $("confidence").textContent = summary.confidence || "-";
  $("calibrated").textContent = visual.calibrated
    ? "yes"
    : "no - score, not probability";
  $("warnings").textContent = (visual.warnings || []).join(", ") || "none";

  renderSignals(visual.signals_detected);

  $("c2pa").textContent = meta.has_c2pa === true
    ? (meta.c2pa_ai_declared ? "present - AI declared" : "present")
    : meta.has_c2pa === false
      ? "absent"
      : "unknown";
  $("software").textContent = meta.software_signature || "Unknown / Stripped";
  $("latency").textContent = meta.latency_ms ? `${meta.latency_ms} ms` : "";

  renderNarrative(data.explanation, data.intent, meta.explanation_provider);

  $("result").classList.remove("hidden");
}

async function detect(file) {
  showError("");
  $("go").disabled = true;
  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(`${config.backendUrl}/detect`, {
      method: "POST",
      headers: authHeaders(),
      body,
    });

    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      // non-JSON body (proxy error page, etc.)
    }

    if (!res.ok) {
      const msg =
        (data && data.error && data.error.message) ||
        (data && data.detail) ||
        `HTTP ${res.status}`;
      const code = data && data.error && data.error.code;
      if (res.status === 401) setStatus("error", "401 - bad API key");
      if (res.status === 503) setStatus("loading", "model loading");
      throw new Error(`${code ? code + ": " : ""}${msg}`);
    }

    renderResult(data);
    checkBackend();
  } catch (e) {
    showError(
      `${e.message} - is BACKEND_URL correct and ALLOWED_ORIGINS on the backend ` +
        `including this site's origin?`
    );
  } finally {
    $("go").disabled = false;
  }
}

// --- wiring -----------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  await fetchEnvConfig();
  applyConfigToInputs();
  readInputsIntoConfig();
  checkBackend();
  setInterval(checkBackend, 15000);

  $("meter").addEventListener("load", flushPendingMeter);
  window.addEventListener("message", (event) => {
    if (event.source !== $("meter").contentWindow) return;
    if (event.data && event.data.type === "rndf:meter-ready") flushPendingMeter();
  });

  $("save").addEventListener("click", saveConfig);

  $("detect-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const file = $("file").files[0];
    if (!file) {
      showError("choose an image first");
      return;
    }
    if (!config.backendUrl) {
      showError("set the backend URL first");
      return;
    }
    detect(file);
  });
});
