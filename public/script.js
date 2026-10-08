/* rndf frontend: fetches the backend URL + key from this deployment's env
 * (/api/config -> BACKEND_URL, DEMO_API_KEY), lets the user override them, then
 * drives the sahu65 backend (Render): GET /readyz for status, POST /detect for
 * predictions. No frameworks - vanilla JS only. */
"use strict";

const $ = (id) => document.getElementById(id);
const LS_URL = "rndf.backendUrl";
const LS_KEY = "rndf.apiKey";

const config = { backendUrl: "", apiKey: "" };

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

function verdictClass(verdict) {
  if (verdict === "likely_ai" || verdict === "ai_generated_verified") return "verdict-ai";
  if (verdict === "likely_real") return "verdict-real";
  return "verdict-inconclusive";
}

function renderResult(data) {
  const verdict = data.verdict || "unknown";
  const p = Number(data.ai_probability ?? 0);

  const v = $("verdict");
  v.textContent = verdict;
  v.className = `verdict ${verdictClass(verdict)}`;

  $("prob-fill").style.width = `${Math.round(p * 100)}%`;
  $("prob-value").textContent = p.toFixed(3);
  $("confidence").textContent = data.confidence || "-";
  $("classifier").textContent =
    data.classifier_probability !== null && data.classifier_probability !== undefined
      ? Number(data.classifier_probability).toFixed(4)
      : "-";
  $("calibrated").textContent = data.calibrated ? "yes" : "no - score, not probability";
  $("warnings").textContent = (data.warnings || []).join(", ") || "none";
  $("latency").textContent = data.latency_ms ? `${data.latency_ms} ms` : "";

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
