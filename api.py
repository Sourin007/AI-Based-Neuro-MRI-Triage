from __future__ import annotations

from functools import lru_cache
import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename

from brain_tumor_ai import BrainTumorWorkflow, Settings


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

app = FastAPI(
    title="Brain Tumor Multi-Agent AI System",
    description="LangGraph-orchestrated diagnostic support workflow for MRI brain tumor triage.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    os.makedirs(settings.uploads_dir, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_workflow() -> BrainTumorWorkflow:
    return BrainTumorWorkflow(get_settings())


def _persist_upload(upload: UploadFile, upload_dir: str) -> str:
    filename = secure_filename(upload.filename or "")
    _, extension = os.path.splitext(filename.lower())
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    saved_name = f"{uuid.uuid4().hex}{extension}"
    destination = os.path.join(upload_dir, saved_name)
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return destination


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Multi-Agent AI system is running.",
        "analyze_endpoint": "/analyze",
        "health_endpoint": "/health",
    }


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "model_path": settings.model_path,
        "knowledge_base_dir": settings.knowledge_base_dir,
    }


@app.post("/analyze")
def analyze_mri(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    settings = get_settings()
    image_path = _persist_upload(file, settings.uploads_dir)

    try:
        workflow = get_workflow()
        state = workflow.invoke(image_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return jsonable_encoder(state["final_report"])
