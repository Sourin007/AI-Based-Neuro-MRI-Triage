from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    base_dir: str
    model_path: str
    knowledge_base_dir: str
    vector_store_dir: str
    uploads_dir: str
    confidence_threshold: float = 0.78
    max_review_loops: int = 2
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    retrieval_k: int = 4
    review_margin_threshold: float = 0.18
    no_tumor_acceptance_threshold: float = 0.82
    meningioma_override_probability: float = 0.22
    meningioma_override_ratio: float = 0.58

    @classmethod
    def from_env(cls, base_dir: str | None = None) -> "Settings":
        project_root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return cls(
            base_dir=project_root,
            model_path=os.path.join(project_root, "models", "model.h5"),
            knowledge_base_dir=os.path.join(project_root, "knowledge_base"),
            vector_store_dir=os.path.join(project_root, ".vector_store"),
            uploads_dir=os.path.join(project_root, "uploads"),
            confidence_threshold=float(os.getenv("BTD_CONFIDENCE_THRESHOLD", "0.78")),
            max_review_loops=int(os.getenv("BTD_MAX_REVIEW_LOOPS", "2")),
            llm_provider=os.getenv("BTD_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("BTD_LLM_MODEL", "gpt-4o-mini"),
            retrieval_k=int(os.getenv("BTD_RETRIEVAL_K", "4")),
            review_margin_threshold=float(os.getenv("BTD_REVIEW_MARGIN_THRESHOLD", "0.18")),
            no_tumor_acceptance_threshold=float(os.getenv("BTD_NO_TUMOR_ACCEPTANCE_THRESHOLD", "0.82")),
            meningioma_override_probability=float(os.getenv("BTD_MENINGIOMA_OVERRIDE_PROBABILITY", "0.22")),
            meningioma_override_ratio=float(os.getenv("BTD_MENINGIOMA_OVERRIDE_RATIO", "0.58")),
        )
