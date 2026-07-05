---
title: Neuro MRI Triage API
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Neuro MRI Triage API

FastAPI + LangGraph diagnostic-support backend for brain-MRI tumor triage
(VGG16 classification, Grad-CAM/LIME explainability, RAG-grounded reporting).

This Space runs the Docker image defined by the repository `Dockerfile`.

## Endpoints
- `GET /health` — liveness
- `GET /ready` — readiness (loads the model + runs a dummy inference)
- `POST /analyze` — multipart `file` upload → full diagnostic report JSON
- `GET /docs` — interactive OpenAPI docs

## Configuration (set under Settings → Variables and secrets)
- `ENVIRONMENT=production`
- `CORS_ALLOW_ORIGINS=https://<your-frontend>.vercel.app`
- `MAX_UPLOAD_MB`, `ANALYZE_RATE_LIMIT`, `SENTRY_DSN`, `OPENAI_API_KEY` (optional)
