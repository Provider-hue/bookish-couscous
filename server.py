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

Endpoints:
    GET  /         service info
    GET  /healthz  liveness (no model involved)
    GET  /readyz   readiness - 200 only once the ~92 MB weights are resident
    POST /detect   multipart image -> sahu65 verdict (Bearer / X-API-Key if API_KEY set)

Deploy: render.yaml runs this file on Render (sahu65 installed from PyPI);
the static frontend in ../public talks to it from Vercel.
"""
import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import sahu65
import uvicorn
from fastapi import FastAPI, File, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

log = logging.getLogger("rndf")


def _load_dotenv() -> None:
    """Merge ./.env (next to this file) into the environment, never overriding
    real process env - Render/Vercel inject their own values."""
    try:
        lines = Path(__file__).resolve().with_name(".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
API_KEY = os.environ.get("API_KEY", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()] or ["*"]
)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ~92 MB ONNX session off the event loop so /readyz flips to ready
    # without the first request paying the ~10 s cost.
    warm_task = asyncio.create_task(asyncio.to_thread(_warm))
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
        result = await asyncio.to_thread(sahu65.detect_bytes, data)
    except Exception as e:
        return _error("bad_image", f"{type(e).__name__}: {e}", 400)
    return result.to_dict()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)
