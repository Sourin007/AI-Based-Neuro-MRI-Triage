from __future__ import annotations

from dataclasses import dataclass
import math

from .state import RiskAssessmentResult


@dataclass(frozen=True)
class RiskThresholds:
    low_to_medium: float
    medium_to_high: float
    high_to_critical: float


TUMOR_TYPE_PRIOR = {
    'notumor': 0.05,
    'pituitary': 0.38,
    'meningioma': 0.52,
    'glioma': 0.78,
}

FEATURE_WEIGHTS = {
    'volume': 0.24,
    'area_ratio': 0.14,
    'spread': 0.18,
    'activation': 0.14,
    'model_confidence': 0.10,
    'tumor_prior': 0.20,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _normalize_volume(volume_cm3: float, reference_cm3: float = 30.0) -> float:
    volume_cm3 = max(volume_cm3, 0.0)
    return _clamp(math.log1p(volume_cm3) / math.log1p(reference_cm3))


def _normalize_ratio(area_ratio: float) -> float:
    return _clamp(area_ratio)


def _compute_explainability_reliability(model_confidence: float, overlap_iou: float | None) -> float:
    iou_component = 0.35 if overlap_iou is None else _clamp(overlap_iou)
    reliability = 0.55 * _clamp(model_confidence) + 0.45 * iou_component
    return _clamp(reliability)


def _adaptive_thresholds(reliability: float, tumor_prior: float) -> RiskThresholds:
    reliability_shift = (0.5 - reliability) * 0.10
    subtype_shift = (tumor_prior - 0.5) * 0.08
    return RiskThresholds(
        low_to_medium=_clamp(0.30 + reliability_shift + subtype_shift, 0.18, 0.42),
        medium_to_high=_clamp(0.55 + reliability_shift + subtype_shift, 0.42, 0.70),
        high_to_critical=_clamp(0.78 + reliability_shift + subtype_shift, 0.65, 0.90),
    )


def compute_risk_score(
    tumor_type: str,
    volume_cm3: float,
    area_ratio: float,
    spread_score: float,
    activation_intensity: float,
    model_confidence: float,
    overlap_iou: float | None,
) -> tuple[float, float, dict[str, float], RiskThresholds]:
    tumor_prior = TUMOR_TYPE_PRIOR.get(tumor_type, 0.5)
    reliability = _compute_explainability_reliability(model_confidence, overlap_iou)

    normalized_features = {
        'volume': _normalize_volume(volume_cm3),
        'area_ratio': _normalize_ratio(area_ratio),
        'spread': _normalize_ratio(spread_score),
        'activation': _normalize_ratio(activation_intensity),
        'model_confidence': _normalize_ratio(model_confidence),
        'tumor_prior': _normalize_ratio(tumor_prior),
    }

    weighted_sum = sum(
        FEATURE_WEIGHTS[name] * normalized_features[name]
        for name in FEATURE_WEIGHTS
    )
    score = _clamp(weighted_sum * (0.70 + 0.30 * reliability))
    thresholds = _adaptive_thresholds(reliability, tumor_prior)
    contributions = {
        name: round(FEATURE_WEIGHTS[name] * normalized_features[name], 4)
        for name in FEATURE_WEIGHTS
    }
    contributions['reliability'] = round(reliability, 4)
    return score, reliability, contributions, thresholds


def categorize_risk(score: float, thresholds: RiskThresholds) -> str:
    if score >= thresholds.high_to_critical:
        return 'critical'
    if score >= thresholds.medium_to_high:
        return 'high'
    if score >= thresholds.low_to_medium:
        return 'medium'
    return 'low'


def build_risk_assessment(
    tumor_type: str,
    volume_cm3: float,
    area_ratio: float,
    spread_score: float,
    activation_intensity: float,
    model_confidence: float,
    overlap_iou: float | None,
    rationale: str,
    recommended_next_steps: list[str],
    urgent_flags: list[str],
) -> RiskAssessmentResult:
    score, reliability, contributions, thresholds = compute_risk_score(
        tumor_type=tumor_type,
        volume_cm3=volume_cm3,
        area_ratio=area_ratio,
        spread_score=spread_score,
        activation_intensity=activation_intensity,
        model_confidence=model_confidence,
        overlap_iou=overlap_iou,
    )
    level = categorize_risk(score, thresholds)
    return RiskAssessmentResult(
        level=level,
        rationale=rationale,
        recommended_next_steps=recommended_next_steps,
        urgent_flags=urgent_flags,
        risk_score=score,
        explainability_reliability=reliability,
        feature_contributions=contributions,
    )
