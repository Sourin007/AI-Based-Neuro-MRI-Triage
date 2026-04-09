# Brain Tumor Multi-Agent AI System

This project now contains a production-style diagnostic support pipeline built around the existing MRI classifier. The classifier is wrapped inside a LangGraph-orchestrated workflow that coordinates explanation, retrieval, risk assessment, critic validation, multi-expert consensus, and structured report generation.

## Architecture

- `brain_tumor_ai/classifier.py`: wraps the existing CNN model and produces binary plus subtype predictions.
- `brain_tumor_ai/rag.py`: loads a local medical knowledge corpus, chunks it, and retrieves supporting context for grounded outputs.
- `brain_tumor_ai/agents.py`: defines the explanation, risk, critic, consensus, and report agents.
- `brain_tumor_ai/graph.py`: defines the shared workflow state and conditional LangGraph routing.
- `api.py`: FastAPI service exposing `/health` and `/analyze`.

## Shared Workflow State

The workflow state includes:
- `input_image`
- `prediction`
- `confidence`
- `explanation`
- `retrieved_context`
- `risk_assessment`
- `validation`
- `consensus`
- `final_report`

## Running Locally

1. Install dependencies from `requirements.txt`.
2. Optionally set `OPENAI_API_KEY` to enable LangChain LLM execution.
3. Start the API with:

```bash
uvicorn api:app --reload