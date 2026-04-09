from __future__ import annotations

from collections import Counter
import os

from .config import Settings
from .risk import build_risk_assessment
from .state import (
    ConsensusResult,
    ExplainabilityResult,
    ExpertOpinion,
    ExplanationResult,
    FinalReport,
    PipelineState,
    RetrievedContextItem,
    RiskAssessmentResult,
    ValidationResult,
)

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    LANGCHAIN_LLM_AVAILABLE = True
except ImportError:
    ChatPromptTemplate = None
    ChatOpenAI = None
    LANGCHAIN_LLM_AVAILABLE = False


class AgentSuite:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = self._build_llm()

    def explanation_agent(self, state: PipelineState) -> ExplanationResult:
        if self.llm is None:
            return self._fallback_explanation(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a cautious medical AI assistant. Explain MRI classifier outputs clearly, do not invent findings, and explicitly state uncertainty and the need for clinician review.",
                ),
                (
                    "human",
                    "Prediction: {prediction}\nConfidence: {confidence:.2f}\nRetrieved context: {retrieved_context}\nReturn a concise explanation with reasoning bullets and caveats.",
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(ExplanationResult)
        return chain.invoke(
            {
                "prediction": state["prediction"],
                "confidence": state["confidence"],
                "retrieved_context": state.get("retrieved_context", []),
            }
        )

    def risk_agent(self, state: PipelineState) -> RiskAssessmentResult:
        if self.llm is None:
            return self._fallback_risk(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You assess risk using model output plus grounded context. Be conservative and avoid diagnosis claims beyond the evidence.",
                ),
                (
                    "human",
                    "Prediction: {prediction}\nConfidence: {confidence:.2f}\nExplainability metrics: {explainability}\nMedical context: {retrieved_context}\nClassify risk as low, medium, high, or critical using lesion size, spread, activation evidence, tumor subtype, and explainability reliability, and recommend next clinical steps.",
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(RiskAssessmentResult)
        return chain.invoke(
            {
                "prediction": state["prediction"],
                "confidence": state["confidence"],
                "explainability": state.get("explainability", {}),
                "retrieved_context": state.get("retrieved_context", []),
            }
        )

    def critic_agent(self, state: PipelineState) -> ValidationResult:
        if self.llm is None:
            return self._fallback_validation(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the critic agent. Reject unsupported conclusions, low-confidence overreach, or reports that lack medical grounding.",
                ),
                (
                    "human",
                    "Prediction: {prediction}\nConfidence: {confidence:.2f}\nExplainability: {explainability}\nExplanation: {explanation}\nRetrieved context: {retrieved_context}\nRisk: {risk_assessment}\nConsensus: {consensus}\nAttempt count: {attempt_count}\nValidate the pipeline output and flag unsupported lesion localization or unjustified risk claims.",
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(ValidationResult)
        result = chain.invoke(
            {
                "prediction": state["prediction"],
                "confidence": state["confidence"],
                "explainability": state.get("explainability", {}),
                "explanation": state.get("explanation"),
                "retrieved_context": state.get("retrieved_context", []),
                "risk_assessment": state.get("risk_assessment"),
                "consensus": state.get("consensus"),
                "attempt_count": state.get("validation", {}).get("attempt_count", 0),
            }
        )
        result.attempt_count = state.get("validation", {}).get("attempt_count", 0) + 1
        return result

    def consensus_agent(self, state: PipelineState) -> ConsensusResult:
        opinions = self._generate_expert_opinions(state)
        decisions = [opinion.decision for opinion in opinions]
        majority_vote = Counter(decisions).most_common(1)[0][0]

        return ConsensusResult(
            summary=f"Consensus based on {len(opinions)} simulated expert roles with majority vote on '{majority_vote}'.",
            final_decision=majority_vote,
            strategy="majority_vote_with_role_confidence_review",
            opinions=opinions,
        )

    def report_agent(self, state: PipelineState) -> FinalReport:
        explanation = ExplanationResult(**state["explanation"])
        context = [RetrievedContextItem(**item) for item in state.get("retrieved_context", [])]
        risk = RiskAssessmentResult(**state["risk_assessment"])
        validation = ValidationResult(**state["validation"])
        consensus = ConsensusResult(**state["consensus"])
        explainability = ExplainabilityResult(**state["explainability"])

        report_text = self._build_report_text(state, explanation, context, risk, validation, consensus)
        return FinalReport(
            prediction=state["prediction"],
            explanation=explanation,
            supporting_evidence=context,
            risk_assessment=risk,
            validation=validation,
            consensus=consensus,
            explainability=explainability,
            report_text=report_text,
            disclaimer=(
                "This AI-generated report is decision support only and must be reviewed by a qualified radiologist or treating physician before any clinical action."
            ),
        )

    def _build_llm(self):
        if not LANGCHAIN_LLM_AVAILABLE:
            return None
        if self.settings.llm_provider != "openai":
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=self.settings.llm_model, temperature=0)

    def _fallback_explanation(self, state: PipelineState) -> ExplanationResult:
        prediction = state["prediction"]
        confidence = state["confidence"]
        subtype = prediction["subtype_label"]
        tumor_flag = prediction["binary_label"]
        summary = (
            f"The classifier favors '{tumor_flag}' with subtype signal '{subtype}' at {confidence:.1%} confidence."
            if tumor_flag == "tumor"
            else f"The classifier favors 'no_tumor' with {confidence:.1%} confidence."
        )
        reasoning = [
            "The explanation reflects the CNN output probabilities rather than a direct radiologist observation.",
            f"The strongest class activation in the current model output is '{subtype}'.",
            "Confidence should be interpreted as model certainty, not confirmed clinical probability.",
        ]
        caveats = [
            "MRI acquisition quality, slice selection, and preprocessing can affect model behavior.",
            "Clinical history, formal radiology review, and additional imaging remain necessary.",
        ]
        return ExplanationResult(summary=summary, reasoning=reasoning, caveats=caveats)

    def _fallback_risk(self, state: PipelineState) -> RiskAssessmentResult:
        confidence = float(state["confidence"])
        prediction = state["prediction"]["binary_label"]
        subtype = state["prediction"]["subtype_label"]
        explainability = state.get("explainability", {})
        volume_cm3 = float(explainability.get("estimated_volume_cm3", 0.0))
        area_ratio = float(explainability.get("tumor_area_ratio", 0.0))
        spread_score = float(explainability.get("spread_score", 0.0))
        activation_intensity = float(explainability.get("activation_intensity", 0.0))
        overlap_iou = explainability.get("overlap_iou")
        overlap_iou = None if overlap_iou is None else float(overlap_iou)

        metric_summary = (
            f"volume {volume_cm3:.2f} cm3, area ratio {area_ratio:.1%}, spread {spread_score:.2f}, "
            f"activation intensity {activation_intensity:.2f}, confidence {confidence:.1%}, IoU {overlap_iou if overlap_iou is not None else 'n/a'}"
        )

        subtype_steps = self._subtype_pathway_steps(subtype, prediction)

        flags: list[str] = []
        if prediction == "tumor":
            flags.append(f"Tumor-class signal detected for subtype '{subtype}'.")
        if confidence < self.settings.confidence_threshold:
            flags.append("Model confidence is below the configured review threshold.")
        if overlap_iou is not None and overlap_iou < 0.10:
            flags.append("Explainability disagreement is high; manual review is recommended.")

        rationale = (
            f"Risk is computed from normalized lesion burden, spread, activation strength, tumor-type prior, model confidence, and Grad-CAM/LIME agreement: {metric_summary}."
        )
        assessment = build_risk_assessment(
            tumor_type=subtype,
            volume_cm3=volume_cm3,
            area_ratio=area_ratio,
            spread_score=spread_score,
            activation_intensity=activation_intensity,
            model_confidence=confidence,
            overlap_iou=overlap_iou,
            rationale=rationale,
            recommended_next_steps=subtype_steps,
            urgent_flags=flags,
        )
        if assessment.level in {"high", "critical"}:
            assessment.urgent_flags.append("Normalized risk score indicates expedited specialty review is warranted.")
        return assessment

    def _subtype_pathway_steps(self, subtype: str, prediction: str) -> list[str]:
        if prediction == "no_tumor" or subtype == "notumor":
            return [
                "Request full neuroradiology review before concluding that no clinically meaningful lesion is present.",
                "If symptoms persist or the MRI is limited, consider contrast-enhanced MRI, interval follow-up imaging, or additional sequences as clinically appropriate.",
                "Correlate with neurologic symptoms, examination findings, and prior imaging before closing the case.",
            ]

        if subtype == "meningioma":
            steps = [
                "Discuss specialist neuroradiology and neurosurgical review to confirm whether the lesion behaves like an extra-axial meningioma and whether surveillance or intervention is more appropriate.",
                "If the lesion is symptomatic, causing mass effect, or appears higher grade, consider workup for surgery; radiation may be considered when residual tumor remains or surgery is not feasible.",
                "Smaller selected lesions may be reviewed for stereotactic radiosurgery when clinically appropriate, with pathology and full-sequence MRI guiding the final plan.",
            ]
            return steps

        if subtype == "glioma":
            steps = [
                "Escalate to neuro-oncology and neurosurgical review, as suspected glioma pathways usually depend on maximal safe resection or biopsy for tissue diagnosis and grading when feasible.",
                "Review contrast-enhanced MRI, edema pattern, and prior imaging to guide whether surgery, radiation, chemotherapy, or combined chemoradiation should be discussed after pathology.",
                "Use multidisciplinary planning rather than AI output alone, because glioma treatment is driven by grade, molecular markers, symptoms, and lesion location.",
            ]
            return steps

        if subtype == "pituitary":
            steps = [
                "Request dedicated sellar review with endocrine correlation and visual assessment if there are headaches, visual symptoms, or hormonal concerns.",
                "Transsphenoidal surgery is commonly reviewed for symptomatic or hormonally active pituitary tumors, while radiation or medical therapy may be considered for residual or specific functional lesions.",
                "Pair imaging with endocrine labs and symptom history before selecting surveillance, surgery, or adjunct treatment.",
            ]
            return steps

        return [
            "Request specialist radiology review before making management decisions.",
            "Consider contrast MRI, interval follow-up imaging, or additional sequences if uncertainty remains.",
            "Correlate imaging findings with symptoms, examination, and clinical history.",
        ]

    def _fallback_validation(self, state: PipelineState) -> ValidationResult:
        issues: list[str] = []
        confidence = state["confidence"]
        attempt_count = state.get("validation", {}).get("attempt_count", 0) + 1

        if confidence < self.settings.confidence_threshold:
            issues.append("Classifier confidence is below the configured review threshold.")
        if not state.get("retrieved_context"):
            issues.append("No retrieved medical context was attached to ground the explanation.")
        if not state.get("explanation"):
            issues.append("Explanation output is missing.")
        if not state.get("risk_assessment"):
            issues.append("Risk assessment output is missing.")

        needs_reprocess = bool(issues) and attempt_count <= self.settings.max_review_loops
        critic_notes = (
            "Re-run retrieval and explanation with emphasis on uncertainty, differential diagnosis, and evidence grounding."
            if issues
            else "Outputs are internally consistent and sufficiently grounded for reporting."
        )
        return ValidationResult(
            is_valid=not issues,
            needs_reprocess=needs_reprocess,
            issues=issues,
            critic_notes=critic_notes,
            attempt_count=attempt_count,
        )

    def _generate_expert_opinions(self, state: PipelineState) -> list[ExpertOpinion]:
        prediction = state["prediction"]["binary_label"]
        subtype = state["prediction"]["subtype_label"]
        confidence = state["confidence"]
        context_summary = "; ".join(item["source"] for item in state.get("retrieved_context", [])) or "local corpus"

        role_templates = {
            "radiologist": f"Imaging pattern is most consistent with '{subtype}' based on model output; correlation with full MRI series is required. Grounding sources: {context_summary}.",
            "oncologist": f"If '{prediction}' is confirmed, treatment planning depends on histology, grade, and symptoms; current AI output alone is not sufficient. Grounding sources: {context_summary}.",
            "general_physician": f"This result should be escalated through standard referral pathways and matched with symptoms and exam findings. Grounding sources: {context_summary}.",
        }

        opinions: list[ExpertOpinion] = []
        for role, opinion in role_templates.items():
            decision = prediction if confidence >= 0.55 else "indeterminate"
            role_confidence = min(max(confidence - 0.05, 0.35), 0.98)
            opinions.append(
                ExpertOpinion(
                    role=role,
                    opinion=opinion,
                    decision=decision,
                    confidence=role_confidence,
                )
            )
        return opinions

    def _build_report_text(
        self,
        state: PipelineState,
        explanation: ExplanationResult,
        context: list[RetrievedContextItem],
        risk: RiskAssessmentResult,
        validation: ValidationResult,
        consensus: ConsensusResult,
    ) -> str:
        evidence = "\n".join(
            f"- {item.source}: {item.content[:220].strip()}..." for item in context
        ) or "- No supporting knowledge chunks were retrieved."
        next_steps = "\n".join(f"- {step}" for step in risk.recommended_next_steps)
        issues = "\n".join(f"- {issue}" for issue in validation.issues) or "- No validation issues detected."

        return (
            "Brain Tumor AI Diagnostic Support Report\n"
            f"Input image: {state['input_image']}\n"
            f"Prediction: {state['prediction']['binary_label']} ({state['prediction']['subtype_label']})\n"
            f"Confidence: {state['confidence']:.1%}\n\n"
            f"Explanation:\n{explanation.summary}\n"
            + "\n".join(f"- {item}" for item in explanation.reasoning)
            + "\n\nSupporting evidence:\n"
            + evidence
            + "\n\nRisk assessment:\n"
            + f"Level: {risk.level}\nRationale: {risk.rationale}\nRecommended next steps:\n{next_steps}\n\n"
            + f"Consensus:\n{consensus.summary}\n"
            + "\n".join(
                f"- {opinion.role}: {opinion.decision} ({opinion.confidence:.1%}) - {opinion.opinion}"
                for opinion in consensus.opinions
            )
            + "\n\nValidation:\n"
            + issues
            + f"\nCritic notes: {validation.critic_notes}\n"
        )
