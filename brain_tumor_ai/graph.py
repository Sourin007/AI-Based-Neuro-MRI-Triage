from __future__ import annotations

from .agents import AgentSuite
from .classifier import BrainTumorClassifier
from .config import Settings
from .explainability import ExplainabilityGenerator
from .rag import MedicalKnowledgeBase
from .state import PipelineState

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class BrainTumorWorkflow:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.classifier = BrainTumorClassifier(self.settings)
        self.explainability = ExplainabilityGenerator()
        self.knowledge_base = MedicalKnowledgeBase(self.settings)
        self.agents = AgentSuite(self.settings)
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def invoke(self, input_image: str) -> PipelineState:
        initial_state: PipelineState = {
            "input_image": input_image,
            "validation": {
                "attempt_count": 0,
                "issues": [],
                "critic_notes": "",
                "is_valid": False,
                "needs_reprocess": False,
            },
        }
        if self.graph is not None:
            return self.graph.invoke(initial_state)
        return self._run_fallback(initial_state)

    def _build_graph(self):
        workflow = StateGraph(PipelineState)
        workflow.add_node("classify_image", self.classify_image)
        workflow.add_node("generate_explainability", self.generate_explainability)
        workflow.add_node("generate_explanation", self.generate_explanation)
        workflow.add_node("retrieve_medical_context", self.retrieve_medical_context)
        workflow.add_node("assess_risk", self.assess_risk)
        workflow.add_node("expert_consensus", self.expert_consensus)
        workflow.add_node("critic_review", self.critic_review)
        workflow.add_node("generate_report", self.generate_report)

        workflow.set_entry_point("classify_image")
        workflow.add_edge("classify_image", "generate_explainability")
        workflow.add_edge("generate_explainability", "generate_explanation")
        workflow.add_edge("generate_explanation", "retrieve_medical_context")
        workflow.add_edge("retrieve_medical_context", "assess_risk")
        workflow.add_edge("assess_risk", "expert_consensus")
        workflow.add_edge("expert_consensus", "critic_review")
        workflow.add_conditional_edges(
            "critic_review",
            self.route_after_critic,
            {
                "retrieve_medical_context": "retrieve_medical_context",
                "generate_report": "generate_report",
            },
        )
        workflow.add_edge("generate_report", END)
        return workflow.compile()

    def classify_image(self, state: PipelineState) -> PipelineState:
        prediction = self.classifier.predict(state["input_image"])
        return {
            "prediction": prediction.model_dump(),
            "confidence": prediction.confidence,
            "retrieval_query": f"brain MRI {prediction.binary_label} {prediction.subtype_label}",
        }

    def generate_explainability(self, state: PipelineState) -> PipelineState:
        label = state["prediction"]["subtype_label"]
        class_index = self.classifier.class_labels.index(label)
        explainability = self.explainability.generate(
            state["input_image"],
            self.classifier,
            class_index,
        )
        return {"explainability": explainability.model_dump()}

    def generate_explanation(self, state: PipelineState) -> PipelineState:
        explanation = self.agents.explanation_agent(state)
        return {"explanation": explanation.model_dump()}

    def retrieve_medical_context(self, state: PipelineState) -> PipelineState:
        validation = state.get("validation", {})
        critic_notes = validation.get("critic_notes", "")
        query = " ".join(part for part in [state.get("retrieval_query", ""), critic_notes] if part).strip()
        context = self.knowledge_base.retrieve(query, k=self.settings.retrieval_k)
        return {
            "retrieval_query": query,
            "retrieved_context": [item.model_dump() for item in context],
        }

    def assess_risk(self, state: PipelineState) -> PipelineState:
        risk = self.agents.risk_agent(state)
        return {"risk_assessment": risk.model_dump()}

    def expert_consensus(self, state: PipelineState) -> PipelineState:
        consensus = self.agents.consensus_agent(state)
        return {"consensus": consensus.model_dump()}

    def critic_review(self, state: PipelineState) -> PipelineState:
        validation = self.agents.critic_agent(state)
        return {"validation": validation.model_dump()}

    def generate_report(self, state: PipelineState) -> PipelineState:
        report = self.agents.report_agent(state)
        return {"final_report": report.model_dump()}

    def route_after_critic(self, state: PipelineState) -> str:
        validation = state.get("validation", {})
        if validation.get("needs_reprocess") and validation.get("attempt_count", 0) <= self.settings.max_review_loops:
            return "retrieve_medical_context"
        return "generate_report"

    def _run_fallback(self, state: PipelineState) -> PipelineState:
        state.update(self.classify_image(state))
        state.update(self.generate_explainability(state))
        state.update(self.generate_explanation(state))
        state.update(self.retrieve_medical_context(state))
        state.update(self.assess_risk(state))
        state.update(self.expert_consensus(state))
        state.update(self.critic_review(state))

        while self.route_after_critic(state) == "retrieve_medical_context":
            state.update(self.retrieve_medical_context(state))
            state.update(self.generate_explanation(state))
            state.update(self.assess_risk(state))
            state.update(self.expert_consensus(state))
            state.update(self.critic_review(state))

        state.update(self.generate_report(state))
        return state
