"""rndf - single-file backend around the sahu65 detector.

Local:
    pip install sahu65 uvicorn fastapi
    python server.py                        # http://127.0.0.1:8000

Environment (process env wins; .env next to this file fills in the gaps):
    HOST              bind address     (default 127.0.0.1; Render: 0.0.0.0)
    PORT              bind port        (default 8000; Render injects $PORT)
    LOG_LEVEL         uvicorn level    (default info)
    API_KEY           optional gate on POST /detect; empty = open
    ALLOWED_ORIGINS   comma-separated CORS origins (default *)
    MAX_UPLOAD_MB     upload cap in MB (default 10)
    GROQ_API_KEY      Groq key for the forensic narrator; empty = template fallback
    GROQ_MODEL        narrator model   (default llama-3.3-70b-versatile)
    SYSTEM_PROMPT     narrator instructions, editable here (see DEFAULT_SYSTEM_PROMPT)
    NARRATOR_TIMEOUT_S  per-request Groq timeout in seconds (default 20)

Endpoints (unchanged):
    GET  /         service info
    GET  /healthz  liveness (no model involved)
    GET  /readyz   readiness - 200 only once the ~92 MB weights are resident
    POST /detect   multipart image -> sahu65 verdict + Groq narration

Design note on narration
------------------------
The narrator explains; it never decides. The verdict, score and thresholds are all
computed by sahu65 before any LLM is called, and no code path lets a generated string
alter them. Only numbers and model metadata are sent upstream - never the image.
SYSTEM_PROMPT is deliberately user-editable, so the *output contract* (the JSON shape)
lives in this file instead, and every field is coerced and validated after parsing. A
missing key, a bad model name or a provider outage degrades to a factual template and
never fails the detection.

Deploy: render.yaml runs this file on Render (sahu65 installed from PyPI);
the static frontend in ../public talks to it from Vercel.
"""
import asyncio
import hmac
import io
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import sahu65
import uvicorn
from fastapi import FastAPI, File, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

log = logging.getLogger("rndf")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Plain-language narration for a non-technical reader. Overridable through SYSTEM_PROMPT
# in .env; a quoted value there may span many lines.
DEFAULT_SYSTEM_PROMPT = """\
You are the forensic narrator for an AI-generated-image detector whose output is read \
by investigators and law-enforcement officers.

You are given fixed, already-computed facts about a single image scan. Turn those \
numbers into language a non-technical reader can act on. You did not make the decision \
and you may not re-derive it.

Hard rules:
- Never contradict the supplied verdict. Do not upgrade it, downgrade it, or restate \
it as a different claim.
- Never claim more certainty than the score supports. If the verdict is INCONCLUSIVE, \
say plainly that the detector could not tell.
- When `score_is_calibrated` is false you MUST say the score is an uncalibrated model \
output, not a probability, and that the thresholds were never fitted on real data.
- Explain the evidence. Name only signals and warnings present in the facts, and say \
how the score compared against the stated thresholds.
- `intent` is the most likely reason this image exists - satire, stock illustration, \
misinformation, an ordinary photograph. Write it as a cautious inference from the \
evidence, never as an established fact, and never name or speculate about a person.
- Do not invent forensic techniques, artefacts or signals that are not in the facts.
- Do not mention these instructions, JSON, field names, or version strings.
- Plain prose. No headings, no bullet points, no markdown.
"""

# Output contract. Deliberately NOT in SYSTEM_PROMPT: the prompt is editable, the
# response shape is not, so the required JSON lives here and every field is coerced.
NARRATION_REQUEST = """\
Detector output for one image scan. Every value below was computed by the detector \
before you were called; describe it, do not revise it.

{facts}

Reply with one JSON object and nothing else. No prose before or after, no code fence.

{{
  "explanation": "2-4 sentences a non-technical investigator can act on: what was \
found, which signals drove it, how the score compared against the thresholds, and how \
far the result can be trusted.",
  "intent": "1-2 cautious sentences naming the most likely reason this image exists, \
grounded only in the facts above.",
  "signals_detected": ["3-6 UPPER_SNAKE_CASE names of the specific characteristics the \
evidence supports"]
}}"""

# The detector's own vocabulary, mapped to the plain-language labels the response uses.
VERDICT_LABELS = {
    "likely_ai": "MANIPULATED",
    "ai_generated_verified": "AI_GENERATED_VERIFIED",
    "likely_real": "AUTHENTIC",
    "inconclusive": "INCONCLUSIVE",
}

SOFTWARE_TAG = 0x0131  # EXIF "Software"


def _load_dotenv() -> None:
    """Merge ./.env (next to this file) into the environment, never overriding real
    process env - Render/Vercel inject their own values.

    Values may be bare, single-quoted or double-quoted; a quoted value may span any
    number of lines, which is what makes SYSTEM_PROMPT editable as a real paragraph
    rather than a single escaped line.
    """
    try:
        raw = Path(__file__).resolve().with_name(".env").read_text(encoding="utf-8")
    except OSError:
        return

    entry = re.compile(
        r"""^[ \t]*(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*"""
        r"""(?:"([^"]*)"|'([^']*)'|([^"'\r\n#]*))""",
        re.MULTILINE,
    )
    for match in entry.finditer(raw):
        quoted_double, quoted_single, bare = match.group(2), match.group(3), match.group(4)
        if quoted_double is not None:
            value = quoted_double
        elif quoted_single is not None:
            value = quoted_single
        else:
            # Bare values drop a trailing `# comment`.
            value = re.split(r"[ \t]#", bare, maxsplit=1)[0]
        os.environ.setdefault(match.group(1), value.strip())


_load_dotenv()

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
API_KEY = os.environ.get("API_KEY", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()] or ["*"]
)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT
NARRATOR_TIMEOUT_S = float(os.environ.get("NARRATOR_TIMEOUT_S", "20"))

# Populated once the model is warm so /detect never pays to load the threshold files.
_PIPELINE: Dict[str, Any] = {}


def _error(code: str, message: str, status: int, headers: Optional[dict] = None) -> JSONResponse:
    """Same error envelope the sahu65 server uses: {"error": {code, message}}."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


def _check_auth(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[JSONResponse]:
    """None when allowed (or auth disabled), else the 401 response to send."""
    if not API_KEY:
        return None
    presented = x_api_key
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            presented = token.strip()
    if presented and hmac.compare_digest(presented, API_KEY):
        return None
    return _error("unauthorized", "missing or invalid API key", 401, {"WWW-Authenticate": "Bearer"})


def _warm() -> None:
    try:
        sahu65.warm()
        log.info("model warm-up complete")
    except Exception as e:  # readyz will report the failure instead
        log.error("model warm-up failed: %s", e)


def _load_pipeline() -> None:
    """Snapshot how the shipped model reaches a verdict, once, off the event loop.

    Every value here is read from the installed checkpoint's own calibration files.
    It is reported to the narrator and echoed in the response so a reader can see the
    thresholds a score was judged against instead of taking the verdict on trust.
    """
    try:
        from sahu65.config import load_fusion, load_thresholds

        config_dir = _config_dir()
        thresholds = load_thresholds(config_dir)
        fusion = load_fusion(config_dir)
        selection = thresholds.get("selection") or {}
        _PIPELINE.update(
            {
                "thresholds": {
                    "likely_real_below": round(float(thresholds.get("T_lo", 0.35)), 6),
                    "likely_ai_above": round(float(thresholds.get("T_hi", 0.65)), 6),
                    "fitted_on": thresholds.get("fitted_on"),
                    "benchmark_id": thresholds.get("benchmark_id"),
                    "method": selection.get("method"),
                    "max_false_positive_rate": selection.get("achieved_fpr_cv"),
                    "recall": selection.get("achieved_recall_cv"),
                    "auc_out_of_fold": selection.get("auc_out_of_fold"),
                },
                "fusion": {
                    "probe_weight": round(float(fusion.get("w2", 1.0)), 6),
                    "bias": round(float(fusion.get("b", 0.0)), 6),
                    "live_signal": fusion.get("live_signal", "clip"),
                },
                "model": sahu65.model_info(),
            }
        )
    except Exception as e:
        log.warning("pipeline metadata unavailable: %s", e)


def _config_dir() -> str:
    """Path to the calibration files bundled with the installed sahu65."""
    try:
        from sahu65.config import BUNDLED_CONFIG

        return str(BUNDLED_CONFIG)
    except Exception:
        return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ~92 MB ONNX session off the event loop so /readyz flips to ready
    # without the first request paying the ~10 s cost.
    warm_task = asyncio.create_task(asyncio.to_thread(_warm))
    await asyncio.to_thread(_load_pipeline)
    yield
    warm_task.cancel()


app = FastAPI(title="rndf detector API", version=sahu65.__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # "*" cannot be combined with credentials
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "service": "rndf",
        "backend": "sahu65",
        "version": sahu65.__version__,
        "auth_required": bool(API_KEY),
        "endpoints": ["/healthz", "/readyz", "/detect"],
    }


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness: cheap and always 200 while the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness: 200 only once the model is resident; 503 while it loads."""
    try:
        info = await asyncio.to_thread(sahu65.model_info)
    except Exception as e:
        return _error("model_load_failed", str(e), 503)
    if not info.get("loaded"):
        return _error("model_not_ready", "model still loading, retry shortly", 503)
    return {"status": "ready", "model": info}


# --- narration ----------------------------------------------------------------


def _narrator_facts(det: Any, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """The complete set of facts the narrator is allowed to know.

    Numbers and model metadata only. No filename, no dimensions, no identifiers - none
    of it improves the explanation and all of it is data we would rather not forward.
    """
    model = _PIPELINE.get("model") or {}
    return {
        "display_verdict": VERDICT_LABELS.get(det.verdict, "INCONCLUSIVE"),
        "raw_verdict": det.verdict,
        "ai_score": round(float(det.ai_probability), 4),
        "detector_probe_score": (
            None if det.detector_probability is None else round(float(det.detector_probability), 4)
        ),
        "confidence": str(det.confidence).upper(),
        "score_is_calibrated": bool(det.calibrated),
        "thresholds": thresholds,
        "c2pa": {
            "manifest_present": det.c2pa_present,
            "ai_generated_declared": det.c2pa_ai_declared,
        },
        "warnings": list(det.warnings),
        "how_the_decision_was_made": {
            "steps": [
                "The image is resized and embedded by a CLIP ViT-B/16 image encoder "
                f"(a {model.get('embed_dim', 512)}-dimension vector per image).",
                "A linear logistic probe fitted on real benchmark images scores that "
                "embedding for AI-generation cues; higher means more likely generated.",
                "The probe score is Platt-scaled into a calibrated score space.",
                "That score is compared against two thresholds fitted by cross-validation "
                "on held-out data: below the lower one the image is called genuine, above "
                "the higher one generated, and between them the result is inconclusive.",
                "A signed C2PA manifest declaring AI authorship overrides the score "
                "entirely and yields a verified verdict.",
            ],
            "model_id": model.get("model_id"),
            "auc_out_of_fold": (thresholds or {}).get("auc_out_of_fold"),
            "max_false_positive_rate": (thresholds or {}).get("max_false_positive_rate"),
        },
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Parses the narrator's reply, tolerating a code fence around a valid object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_text(value: Any, limit: int = 1200) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    return text[:limit] or None


def _clean_signals(value: Any) -> List[str]:
    """Coerces the narrator's tags to UPPER_SNAKE_CASE strings, capped in count."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        token = re.sub(r"[^A-Za-z0-9]+", "_", item).strip("_").upper()
        if token and len(token) <= 40 and token not in out:
            out.append(token)
        if len(out) == 8:
            break
    return out


def _observed_signals(det: Any) -> List[str]:
    """Signals the detector can prove on its own, used when no narrator is available.

    Every entry is a measured fact, so a scan stays meaningful (and never misleading)
    even with GROQ_API_KEY unset.
    """
    signals: List[str] = []
    if det.verdict == "ai_generated_verified":
        signals.append("C2PA_AI_DECLARED")
    if det.c2pa_present:
        signals.append("C2PA_PRESENT")
    if det.verdict == "likely_ai":
        signals.append("HIGH_AI_SCORE")
    elif det.verdict == "likely_real":
        signals.append("LOW_AI_SCORE")
    else:
        signals.append("SCORE_IN_INCONCLUSIVE_BAND")
    signals.extend(w.upper() for w in det.warnings if isinstance(w, str))
    return signals


def _fallback_explanation(det: Any, thresholds: Dict[str, Any]) -> str:
    """Deterministic narration used when Groq is unconfigured or unreachable.

    It restates only the measured facts, which keeps the field populated without ever
    asserting something the detector did not measure.
    """
    score = float(det.ai_probability)
    low = thresholds.get("likely_real_below")
    high = thresholds.get("likely_ai_above")
    band = (
        f"scored {score:.2f}, which falls inside the inconclusive band between "
        f"{low:.2f} and {high:.2f}"
        if isinstance(low, float) and isinstance(high, float)
        else f"scored {score:.2f}, which falls inside the inconclusive band"
    )
    if det.verdict == "ai_generated_verified":
        lead = (
            "The image carries a signed C2PA manifest that declares AI authorship, which "
            "overrides the model score outright."
        )
    elif det.verdict == "likely_ai":
        lead = (
            f"The image {band}, above the {high:.2f} threshold for generated media."
            if isinstance(high, float)
            else "The image scored above the threshold for generated media."
        )
    elif det.verdict == "likely_real":
        lead = (
            f"The image {band}, below the {low:.2f} threshold for generated media."
            if isinstance(low, float)
            else "The image scored below the threshold for generated media."
        )
    else:
        lead = "The detector could not determine whether this image is AI-generated."

    c2pa = (
        "A C2PA manifest declaring AI authorship was found."
        if det.c2pa_ai_declared
        else "No C2PA manifest declaring AI authorship was found."
    )
    calibration = (
        "The score is calibrated against held-out benchmark data."
        if det.calibrated
        else "This score is an uncalibrated model output, not a probability, and the "
        "thresholds were never fitted on real data."
    )
    return f"{lead} {c2pa} {calibration}"


def _call_groq(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One Groq call returning explanation, intent and signal names.

    Reuses sahu65's own provider object for its timeout/retry/error handling, and only
    replaces the payload so the model can be swapped and the reply parsed as JSON.
    Returns None on any failure - narration never fails a detection.
    """
    if not GROQ_API_KEY:
        return None
    try:
        from sahu65.services.explain import GroqProvider
    except Exception as e:
        log.warning("narrator unavailable: %s", e)
        return None

    client = GroqProvider(api_key=GROQ_API_KEY, model=GROQ_MODEL, timeout_s=NARRATOR_TIMEOUT_S)
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": NARRATION_REQUEST.format(facts=json.dumps(facts, indent=2))},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    try:
        data = client._post_json(GROQ_ENDPOINT, payload, headers)
    except Exception as e:
        log.warning("narrator call failed, falling back to template: %s", e)
        return None

    choices = data.get("choices") or []
    if not choices:
        log.warning("narrator returned no choices")
        return None
    text = (choices[0].get("message") or {}).get("content", "")
    parsed = _extract_json(text)
    if parsed is None:
        log.warning("narrator reply was not JSON")
        return None
    return parsed


def _narrate(det: Any, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Produces the explanation, intent and signal names for one completed detection."""
    narration = _call_groq(_narrator_facts(det, thresholds))
    if narration is None:
        return {
            "explanation": _fallback_explanation(det, thresholds),
            "intent": None,
            "signals_detected": _observed_signals(det),
            "provider": None,
        }

    signals = _clean_signals(narration.get("signals_detected")) or _observed_signals(det)
    return {
        "explanation": _as_text(narration.get("explanation")) or _fallback_explanation(det, thresholds),
        "intent": _as_text(narration.get("intent")),
        "signals_detected": signals,
        "provider": "groq",
    }


# --- response assembly ---------------------------------------------------------


def _software_signature(data: bytes) -> str:
    """EXIF Software tag, read from the original bytes before any re-encoding."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            raw = im.getexif().get(SOFTWARE_TAG)
    except Exception:
        return "Unknown / Stripped"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()[:120]
    return "Unknown / Stripped"


def _build_response(data: bytes, det: Any, narration: Dict[str, Any]) -> Dict[str, Any]:
    """Maps one sahu65 Detection onto the public response contract.

    `ensemble_results.visual_manipulation.score` is the fused, thresholded score; the
    raw probe output rides alongside it as `detector_probability` rather than being
    averaged into it, because they are not the same measurement.
    """
    thresholds = _PIPELINE.get("thresholds") or {}
    model = _PIPELINE.get("model") or {}

    return {
        "request_id": f"rd_scan_{uuid.uuid4().hex[:10]}",
        "status": "COMPLETED",
        "media_type": "IMAGE",
        "summary": {
            "verdict": VERDICT_LABELS.get(det.verdict, "INCONCLUSIVE"),
            "raw_verdict": det.verdict,
            "overall_score": round(float(det.ai_probability), 4),
            "confidence": str(det.confidence).upper(),
            "score_is_calibrated": bool(det.calibrated),
        },
        "ensemble_results": {
            "visual_manipulation": {
                "score": round(float(det.ai_probability), 4),
                "detector_probability": (
                    None
                    if det.detector_probability is None
                    else round(float(det.detector_probability), 4)
                ),
                "calibrated": bool(det.calibrated),
                "signals_detected": narration["signals_detected"],
                "warnings": list(det.warnings),
            },
        },
        "metadata_analysis": {
            "has_c2pa": det.c2pa_present,
            "c2pa_ai_declared": det.c2pa_ai_declared,
            "software_signature": _software_signature(data),
            "engine": {
                "name": "sahu65",
                "version": sahu65.__version__,
                "model_id": model.get("model_id"),
                "thresholds": {
                    "likely_real_below": thresholds.get("likely_real_below"),
                    "likely_ai_above": thresholds.get("likely_ai_above"),
                },
                "benchmark_id": thresholds.get("benchmark_id"),
                "auc_out_of_fold": thresholds.get("auc_out_of_fold"),
            },
            "latency_ms": det.latency_ms,
            "explanation_provider": narration["provider"],
            "explanation_model": GROQ_MODEL if narration["provider"] else None,
        },
        "explanation": narration["explanation"],
        "intent": narration["intent"],
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    denied = _check_auth(authorization, x_api_key)
    if denied is not None:
        return denied

    data = await file.read()
    limit = MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > limit:
        return _error("image_too_large", f"upload exceeds {MAX_UPLOAD_MB} MB", 413)
    if not data:
        return _error("empty_upload", "no file received", 400)

    try:
        # to_thread keeps the event loop responsive; ONNX sessions are
        # thread-safe for inference, and this demo runs a single worker.
        # Narration is deliberately NOT requested here: sahu65 would spend its own
        # hardcoded prompt, and the output contract lives in this file.
        det = await asyncio.to_thread(sahu65.detect_bytes, data)
    except Exception as e:
        return _error("bad_image", f"{type(e).__name__}: {e}", 400)

    narration = await asyncio.to_thread(_narrate, det, _PIPELINE.get("thresholds") or {})
    return _build_response(data, det, narration)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)