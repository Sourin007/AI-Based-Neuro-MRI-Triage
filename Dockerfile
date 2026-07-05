# syntax=docker/dockerfile:1

##########  Builder: install deps into an isolated venv  ##########
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tools kept out of the final image (needed only if a wheel must compile).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


##########  Runtime: slim image with only the venv + app code  ##########
FROM python:3.10-slim AS runtime

# libgomp1 = OpenMP runtime required by numpy/scipy/scikit-learn/tensorflow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Application code + runtime assets (model weights + RAG knowledge base baked in).
COPY --chown=appuser:appuser brain_tumor_ai/ ./brain_tumor_ai/
COPY --chown=appuser:appuser knowledge_base/ ./knowledge_base/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser app_config.py logging_config.py api.py ./

# Writable dirs created fresh in the image (excluded from the build context).
RUN mkdir -p /app/uploads /app/.vector_store && chown -R appuser:appuser /app

USER appuser

# HF Spaces defaults to 7860; Cloud Run / others inject $PORT.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
