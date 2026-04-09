from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


TumorSubtype = Literal["pituitary", "glioma", "notumor", "meningioma"]
RiskLevel = Literal["low", "medium", "high", "critical"]


class PredictionResult(BaseModel):
    binary_label: Literal["tumor", "no_tumor"]
    subtype_label: TumorSubtype
    confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    raw_subtype_label: TumorSubtype
    raw_confidence: float = Field(ge=0.0, le=1.0)
    review_recommended: bool = False
    decision_status: Literal["accepted", "review_required"] = "accepted"
    correction_reason: str | None = None


class ExplanationResult(BaseModel):
    summary: str
    reasoning: list[str]
    caveats: list[str]


class RetrievedContextItem(BaseModel):
    source: str
    content: str
    relevance_score: float


class RiskAssessmentResult(BaseModel):
    level: RiskLevel
    rationale: str
    recommended_next_steps: list[str]
    urgent_flags: list[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    explainability_reliability: float = Field(ge=0.0, le=1.0)
    feature_contributions: dict[str, float] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    is_valid: bool
    needs_reprocess: bool
    issues: list[str]
    critic_notes: str
    attempt_count: int = 0


class ExpertOpinion(BaseModel):
    role: Literal["radiologist", "oncologist", "general_physician"]
    opinion: str
    decision: Literal["tumor", "no_tumor", "indeterminate"]
    confidence: float = Field(ge=0.0, le=1.0)


class ConsensusResult(BaseModel):
    summary: str
    final_decision: Literal["tumor", "no_tumor", "indeterminate"]
    strategy: str
    opinions: list[ExpertOpinion]


class ExplainabilityResult(BaseModel):
    uploaded_image: str
    gradcam_overlay: str
    tumor_mask_overlay: str
    lime_overlay: str | None = None
    gradcam_peak: dict[str, float]
    lime_available: bool
    notes: list[str] = Field(default_factory=list)
    tumor_area_px: int = Field(ge=0)
    tumor_area_ratio: float = Field(ge=0.0, le=1.0)
    estimated_volume_cm3: float = Field(ge=0.0)
    activation_intensity: float = Field(ge=0.0, le=1.0)
    spread_score: float = Field(ge=0.0, le=1.0)
    overlap_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    pixel_spacing_mm: float = Field(gt=0.0)
    slice_thickness_mm: float = Field(gt=0.0)


class FinalReport(BaseModel):
    prediction: PredictionResult
    explanation: ExplanationResult
    supporting_evidence: list[RetrievedContextItem]
    risk_assessment: RiskAssessmentResult
    validation: ValidationResult
    consensus: ConsensusResult
    explainability: ExplainabilityResult
    report_text: str
    disclaimer: str


class PipelineState(TypedDict, total=False):
    input_image: str
    prediction: dict[str, Any]
    confidence: float
    explanation: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    retrieval_query: str
    risk_assessment: dict[str, Any]
    validation: dict[str, Any]
    consensus: dict[str, Any]
    explainability: dict[str, Any]
    final_report: dict[str, Any]
