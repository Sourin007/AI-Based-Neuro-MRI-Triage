from functools import lru_cache
import io
import os
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
from werkzeug.utils import secure_filename

from app_config import AppConfig
from brain_tumor_ai import BrainTumorWorkflow, Settings
from logging_config import configure_logging, get_logger, init_sentry


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# --- Startup configuration (fails fast on invalid config) --------------------
config = AppConfig.from_env()
configure_logging(log_level=config.log_level, json_logs=config.json_logs)
logger = get_logger("api")
_sentry_enabled = init_sentry(config.sentry_dsn, config.environment)
logger.info(
    "startup",
    environment=config.environment,
    cors_origins=list(config.cors_allow_origins),
    max_upload_mb=round(config.max_upload_bytes / 1024 / 1024, 2),
    rate_limit=config.analyze_rate_limit,
    sentry_enabled=_sentry_enabled,
)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Brain Tumor Multi-Agent AI System",
    description="LangGraph-orchestrated diagnostic support workflow for MRI brain tumor triage.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: _rate_limited(request, exc))


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.cors_allow_origins),
    allow_origin_regex=config.cors_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _rate_limited(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse

    logger.warning("rate_limited", path=request.url.path, client=get_remote_address(request))
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    os.makedirs(settings.uploads_dir, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_workflow() -> BrainTumorWorkflow:
    return BrainTumorWorkflow(get_settings())


@lru_cache(maxsize=1)
def _warm_model() -> bool:
    """Load the model and run one dummy inference. Cached on success; retried on failure."""
    classifier = get_workflow().classifier
    size = classifier.image_size
    dummy = np.zeros((size, size, 3), dtype="float32")
    batch = np.expand_dims(classifier.preprocess_array(dummy), axis=0)
    classifier.model.predict(batch, verbose=0)
    return True


def _validate_and_read(upload: UploadFile, max_bytes: int) -> tuple[bytes, str]:
    """Enforce size + image-content validity. Returns (raw_bytes, extension)."""
    filename = secure_filename(upload.filename or "")
    _, extension = os.path.splitext(filename.lower())
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    # Read one byte past the limit to detect oversized uploads without loading everything.
    raw = upload.file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Confirm the bytes are actually a decodable image, not just a matching extension.
    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    return raw, extension


def _persist_bytes(raw: bytes, extension: str, upload_dir: str) -> str:
    saved_name = f"{uuid.uuid4().hex}{extension}"
    destination = os.path.join(upload_dir, saved_name)
    with open(destination, "wb") as buffer:
        buffer.write(raw)
    return destination


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Multi-Agent AI system is running.",
        "analyze_endpoint": "/analyze",
        "health_endpoint": "/health",
        "readiness_endpoint": "/ready",
    }


@app.get("/health")
def health():
    """Liveness: the process is up. Does not check the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: the model is loaded and can run inference."""
    try:
        _warm_model()
    except Exception:
        logger.error("readiness_failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Model not ready.")
    settings = get_settings()
    return {
        "status": "ready",
        "model_path": settings.model_path,
        "knowledge_base_dir": settings.knowledge_base_dir,
    }


@app.post("/analyze")
@limiter.limit(config.analyze_rate_limit)
def analyze_mri(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    settings = get_settings()
    raw, extension = _validate_and_read(file, config.max_upload_bytes)
    image_path = _persist_bytes(raw, extension, settings.uploads_dir)
    logger.info("analyze_start", bytes=len(raw))

    try:
        workflow = get_workflow()
        state = workflow.invoke(image_path)
    except Exception:
        # Full detail server-side only; clients get a generic message + X-Request-ID header.
        logger.error("analysis_failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again; contact support with the request ID if it persists.",
        )

    logger.info("analyze_success")
    return jsonable_encoder(state["final_report"])
