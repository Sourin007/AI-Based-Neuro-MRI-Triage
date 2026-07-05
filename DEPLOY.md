# Deployment Guide

Backend → **Hugging Face Spaces** (Docker). Frontend → **Vercel**.
The two are wired together at the end via `VITE_API_BASE_URL` (frontend → backend)
and `CORS_ALLOW_ORIGINS` (backend → frontend).

---

## Part 1 — Backend on Hugging Face Spaces

### 1.1  Create the account, Space, and token  (YOU do this)
1. Sign up / log in at **https://huggingface.co**.
2. Create a Space: **https://huggingface.co/new-space**
   - Owner: your username
   - Space name: e.g. `neuro-mri-triage-api`
   - License: your choice
   - **Space SDK: Docker → "Blank"**
   - Hardware: **CPU basic (free)**
   - Visibility: Public
   - Click **Create Space**. Copy the Space URL (e.g. `https://huggingface.co/spaces/<user>/neuro-mri-triage-api`).
3. Create a write token: **https://huggingface.co/settings/tokens** →
   **New token** → type **Write** → copy it.
4. In your terminal, authenticate once (keeps the token with you, not shared):
   ```
   huggingface-cli login
   ```
   Paste the token when prompted. (`huggingface-cli` ships with the project deps.)

### 1.2  Push code + model to the Space  (I run this once you give me the Space URL)
Assembles a Space working copy, LFS-tracks the 164 MB model, pushes. HF then
builds the Dockerfile automatically.
```
git clone https://huggingface.co/spaces/<user>/<space>  ../hf-space
cd ../hf-space
git lfs install
cp -r ../AI-Based-Neuro-MRI-Triage/brain_tumor_ai ../AI-Based-Neuro-MRI-Triage/knowledge_base ../AI-Based-Neuro-MRI-Triage/models .
cp ../AI-Based-Neuro-MRI-Triage/api.py ../AI-Based-Neuro-MRI-Triage/app_config.py ../AI-Based-Neuro-MRI-Triage/logging_config.py ../AI-Based-Neuro-MRI-Triage/requirements.txt ../AI-Based-Neuro-MRI-Triage/Dockerfile ../AI-Based-Neuro-MRI-Triage/.dockerignore .
cp ../AI-Based-Neuro-MRI-Triage/deploy/hf_space_README.md ./README.md
cp ../AI-Based-Neuro-MRI-Triage/deploy/gitattributes_for_hf ./.gitattributes
git add -A && git commit -m "Deploy Neuro MRI Triage backend" && git push
```

### 1.3  Configure environment  (YOU, in the Space UI)
Space → **Settings → Variables and secrets** → add:
- `ENVIRONMENT=production`
- `CORS_ALLOW_ORIGINS=` *(leave blank for now — filled after Vercel; until then keep `ENVIRONMENT=development` so the app boots without it)*
- optional: `SENTRY_DSN`, `OPENAI_API_KEY`, `MAX_UPLOAD_MB`, `ANALYZE_RATE_LIMIT`

### 1.4  Verify
Watch the **build logs** in the Space. When it shows "Running", the public URL is
`https://<user>-<space>.hf.space`. Test:
- `https://<user>-<space>.hf.space/health` → `{"status":"ok"}`
- `https://<user>-<space>.hf.space/docs` → upload an MRI, run `/analyze`

---

## Part 2 — Frontend on Vercel

### 2.1  (YOU) Create a Vercel account at https://vercel.com and connect your GitHub.
### 2.2  Import the repo, set **Root Directory = `neuro-ui`**.
### 2.3  Add an env var `VITE_API_BASE_URL = https://<user>-<space>.hf.space`.
### 2.4  Deploy. Vercel gives a URL like `https://<project>.vercel.app`.

---

## Part 3 — Wire them together
1. Go back to the HF Space → set `CORS_ALLOW_ORIGINS=https://<project>.vercel.app`
   and `ENVIRONMENT=production`. The Space restarts.
2. Open the Vercel URL, upload an MRI end-to-end. Done.

---

## Redeploying later
- **Backend:** re-copy changed files into `../hf-space` and `git push` (§1.2).
- **Frontend:** push to your GitHub `main` — Vercel auto-redeploys.
