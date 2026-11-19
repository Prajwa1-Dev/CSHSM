"""
CSHSM Milestone 12: Main Pipeline Orchestrator.

Flow:
  Input Text
      ↓
  1. Input Validation (Empty / Whitespace / Non-string check)
      ↓
  2. Text Preprocessing (Conservative cleaning via TextCleaner)
      ↓
  3. Hate Speech Classification (XLM-R / Mock with fail-closed safety)
      ↓
  [If Non-Hate] ─────────→ 4. Non-Hate Early Exit (No counter-speech generated)
      ↓ [If Hate]
  5. Top-K Vector Retrieval & Threshold Routing (M7/M8)
      ├── [Similarity >= tau] ─→ 6a. M11 Final Gate on Retrieved Candidate ─→ RETRIEVED_RESPONSE
      │                                       │ (if safety rejected)
      │                                       └──→ SAFE_FALLBACK
      └── [Similarity < tau] ──→ 6b. M9 Generation Fan-Out (5 models)
                                              ↓
                                 7. Candidate Structural & Safety Filtering
                                              ↓
                                 8. M10 Multi-Criteria Evaluation & Ranking
                                              ↓
                                 9. M11 Final Safety & Validation Gate
                                              ├── [Pass] ─→ GENERATED_RESPONSE
                                              └── [Fail] ─→ SAFE_FALLBACK
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field

from src.data.cleaner import TextCleaner
from src.retrieval.embedder import MockEmbedder
from src.retrieval.retriever import (
    CSHSMRetriever,
    ExactInnerProductIndex,
    RetrievalMatch
)
from src.routing.router import FallbackRouter, RoutingDecision
from src.generation.fan_out import (
    GenerationFanOut,
    GenerationRequest,
    FanOutResult,
    create_default_fan_out
)
from src.evaluation.evaluator import MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker, RankingResult
from src.safety.final_gate import FinalSafetyGate, FinalGateResult
from src.pipeline.classifier import BaseClassifier, MockClassifier, ClassifierOutput


# =====================================================================
# Pipeline Result Data Model
# =====================================================================

@dataclass
class PipelineResult:
    """
    Standardized, auditable outcome of the CSHSM end-to-end pipeline.

    Outcomes:
      - "NON_HATE_EARLY_EXIT": Input classified as non-hate; early exit; no counter-speech.
      - "RETRIEVED_RESPONSE": Input classified as hate; high similarity match passed M11 gate.
      - "GENERATED_RESPONSE": Input classified as hate; low similarity; M9 generated, M10 ranked, M11 passed.
      - "SAFE_FALLBACK": Any safety rejection, failure, empty input, or classifier error.
    """
    outcome: str
    input_text: str
    response_text: Optional[str]
    candidate_id: Optional[str]
    source_origin: Optional[str]  # "RETRIEVAL" | "GENERATION" | "NONE"
    is_hate: bool
    category: Optional[str]
    confidence: float
    model_id: Optional[str]
    provider: Optional[str]
    quality_status: str  # "filtered_candidate" | "generated_candidate" | "non_hate_exit" | "fallback"
    review_status: str   # Always "unreviewed"
    human_audited: bool  # Always False
    safe_fallback_triggered: bool
    fallback_reason: Optional[str]
    fallback_response: Optional[str]
    component_statuses: Dict[str, str]
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Main Pipeline Orchestrator
# =====================================================================

class PipelineOrchestrator:
    """
    CSHSM Milestone 12 Pipeline Orchestrator.

    Integrates:
      - Cleaner: TextCleaner
      - Classifier: BaseClassifier
      - Retriever & Router: CSHSMRetriever & FallbackRouter
      - Fan-Out: GenerationFanOut (5 adapters)
      - Ranker: CandidateRanker
      - Final Gate: FinalSafetyGate

    Invariants:
      - Fail-closed: Catches any unhandled exception and yields safe fallback.
      - Never marks unapproved candidates as approved (review_status="unreviewed").
      - Never performs counter-speech generation on non-hate inputs.
      - Defense-in-depth: Evaluates M11 safety filter on both retrieved and generated outputs.
      - Mock guard: Rejects simulated evidence when allow_mock=False.
    """

    DEFAULT_FALLBACK_MESSAGE = "I cannot respond to this respectfully."
    DISCLAIMER_UNAPPROVED = (
        "VALIDATED UNAPPROVED RESPONSE — Output passed CSHSM pipeline safety gates. "
        "Unapproved for production release pending human review."
    )
    DISCLAIMER_NON_HATE = (
        "NON-HATE INPUT — Early exit; no counter-speech generated."
    )
    DISCLAIMER_FALLBACK = (
        "SAFE FALLBACK — Pipeline triggered safe fallback; candidate rejected or unavailable."
    )

    def __init__(
        self,
        classifier: BaseClassifier,
        router: FallbackRouter,
        fan_out: GenerationFanOut,
        ranker: CandidateRanker,
        final_gate: FinalSafetyGate,
        cleaner: Optional[TextCleaner] = None,
        allow_mock: bool = True,
        threshold: Optional[float] = None,
        fallback_message: str = DEFAULT_FALLBACK_MESSAGE
    ):
        if classifier is None:
            raise ValueError("classifier cannot be None")
        if router is None:
            raise ValueError("router cannot be None")
        if fan_out is None:
            raise ValueError("fan_out cannot be None")
        if ranker is None:
            raise ValueError("ranker cannot be None")
        if final_gate is None:
            raise ValueError("final_gate cannot be None")

        self.classifier = classifier
        self.router = router
        self.fan_out = fan_out
        self.ranker = ranker
        self.final_gate = final_gate
        self.cleaner = cleaner
        self.allow_mock = allow_mock
        self.threshold = threshold
        self.fallback_message = fallback_message

    def process(self, text: str) -> PipelineResult:
        """
        Executes end-to-end processing for a user input text.
        Guaranteed to never raise an unhandled exception to the caller.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        component_statuses: Dict[str, str] = {
            "cleaner": "PENDING",
            "classifier": "PENDING",
            "router": "PENDING",
            "generation": "PENDING",
            "ranking": "PENDING",
            "final_gate": "PENDING"
        }
        provenance: Dict[str, Any] = {
            "pipeline_version": "1.0",
            "start_time": now_iso,
            "allow_mock": self.allow_mock
        }

        try:
            # -------------------------------------------------------------
            # 1. Input Validation
            # -------------------------------------------------------------
            if text is None or not isinstance(text, str) or not text.strip():
                component_statuses["cleaner"] = "EMPTY_INPUT"
                return self._create_fallback_result(
                    input_text=str(text) if text is not None else "",
                    is_hate=False,
                    category=None,
                    confidence=0.0,
                    reason="EMPTY_INPUT: Input text is empty, None, or whitespace only.",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="NONE"
                )

            # -------------------------------------------------------------
            # 2. Text Preprocessing
            # -------------------------------------------------------------
            if self.cleaner is not None:
                cleaned_text = self.cleaner.clean_text(text)
                component_statuses["cleaner"] = "SUCCESS"
            else:
                cleaned_text = text.strip()
                component_statuses["cleaner"] = "PASSTHROUGH"

            if not cleaned_text.strip():
                component_statuses["cleaner"] = "EMPTY_AFTER_CLEANING"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=False,
                    category=None,
                    confidence=0.0,
                    reason="EMPTY_AFTER_CLEANING: Text was empty after normalization.",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="NONE"
                )

            provenance["cleaned_text"] = cleaned_text

            # -------------------------------------------------------------
            # 3. Hate Speech Classification
            # -------------------------------------------------------------
            try:
                cls_out: ClassifierOutput = self.classifier.predict(cleaned_text)
                component_statuses["classifier"] = "SUCCESS"
            except Exception as exc:
                component_statuses["classifier"] = f"ERROR: {type(exc).__name__}"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=False,
                    category=None,
                    confidence=0.0,
                    reason=f"CLASSIFIER_ERROR: {exc}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="NONE"
                )

            # Check mock policy for classifier
            if cls_out.is_mock and not self.allow_mock:
                component_statuses["classifier"] = "MOCK_PROHIBITED"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=cls_out.is_hate,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason="MOCK_PROHIBITED: Mock classifier evidence is prohibited when allow_mock=False.",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="NONE"
                )

            provenance["classifier_output"] = cls_out.to_dict()

            # -------------------------------------------------------------
            # 4. Non-Hate Early Exit
            # -------------------------------------------------------------
            if not cls_out.is_hate:
                component_statuses["router"] = "SKIPPED"
                component_statuses["generation"] = "SKIPPED"
                component_statuses["ranking"] = "SKIPPED"
                component_statuses["final_gate"] = "SKIPPED"
                return PipelineResult(
                    outcome="NON_HATE_EARLY_EXIT",
                    input_text=text,
                    response_text=None,
                    candidate_id=None,
                    source_origin="NONE",
                    is_hate=False,
                    category=None,
                    confidence=cls_out.confidence,
                    model_id=cls_out.model_id,
                    provider="classifier",
                    quality_status="non_hate_exit",
                    review_status="unreviewed",
                    human_audited=False,
                    safe_fallback_triggered=False,
                    fallback_reason=None,
                    fallback_response=None,
                    component_statuses=component_statuses,
                    provenance=provenance,
                    disclaimer=self.DISCLAIMER_NON_HATE
                )

            # -------------------------------------------------------------
            # 5. Hate Input Confirmed -> M7 Retrieval & M8 Routing
            # -------------------------------------------------------------
            try:
                routing_dec: RoutingDecision = self.router.route_text(
                    query_text=cleaned_text,
                    threshold=self.threshold,
                    category_filter=cls_out.category
                )
                component_statuses["router"] = routing_dec.action
            except Exception as exc:
                component_statuses["router"] = f"ERROR: {type(exc).__name__}"
                routing_dec = RoutingDecision(
                    query=cleaned_text,
                    action="FALLBACK_GENERATION",
                    threshold=self.threshold or self.router.default_threshold,
                    similarity_score=None,
                    retrieved_candidate=None,
                    fallback_reason=f"ROUTER_EXCEPTION: {exc}",
                    fallback_payload={"error": str(exc)},
                    provenance={"error": str(exc)},
                    disclaimer=self.DISCLAIMER_FALLBACK
                )

            provenance["routing_decision"] = routing_dec.to_dict()

            # -------------------------------------------------------------
            # 6. Branch A: High Similarity -> M11 Retrieved Candidate Gate
            # -------------------------------------------------------------
            if routing_dec.action == "RETRIEVE" and routing_dec.retrieved_candidate is not None:
                component_statuses["generation"] = "SKIPPED"
                component_statuses["ranking"] = "SKIPPED"
                ret_match = routing_dec.retrieved_candidate

                gate_res = self.final_gate.validate_retrieved_candidate(
                    retrieval_match=ret_match,
                    routing_decision=routing_dec,
                    allow_mock=self.allow_mock
                )
                component_statuses["final_gate"] = gate_res.decision
                provenance["final_gate_result"] = gate_res.to_dict()

                if gate_res.decision == "PASS_UNAPPROVED":
                    return PipelineResult(
                        outcome="RETRIEVED_RESPONSE",
                        input_text=text,
                        response_text=gate_res.text,
                        candidate_id=gate_res.candidate_id,
                        source_origin="RETRIEVAL",
                        is_hate=True,
                        category=cls_out.category,
                        confidence=cls_out.confidence,
                        model_id=gate_res.model_id,
                        provider=gate_res.provider,
                        quality_status=gate_res.quality_status,  # "filtered_candidate"
                        review_status="unreviewed",
                        human_audited=False,
                        safe_fallback_triggered=False,
                        fallback_reason=None,
                        fallback_response=None,
                        component_statuses=component_statuses,
                        provenance=provenance,
                        disclaimer=gate_res.disclaimer
                    )
                else:
                    # Defense-in-depth: M11 safety filter rejected retrieved candidate
                    return self._create_fallback_result(
                        input_text=text,
                        is_hate=True,
                        category=cls_out.category,
                        confidence=cls_out.confidence,
                        reason=f"RETRIEVED_SAFETY_REJECTED: {', '.join(gate_res.failure_reasons)}",
                        component_statuses=component_statuses,
                        provenance=provenance,
                        source_origin="RETRIEVAL",
                        candidate_id=gate_res.candidate_id
                    )

            # -------------------------------------------------------------
            # 7. Branch B: M9 Generation Fan-Out
            # -------------------------------------------------------------
            try:
                fan_out_res: FanOutResult = self.fan_out.execute(
                    routing_dec,
                    category_override=cls_out.category
                )
                component_statuses["generation"] = fan_out_res.status
            except Exception as exc:
                component_statuses["generation"] = f"ERROR: {type(exc).__name__}"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason=f"FAN_OUT_ERROR: {exc}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="GENERATION"
                )

            provenance["fan_out_result"] = fan_out_res.to_dict()

            if len(fan_out_res.safe_candidates) == 0:
                component_statuses["ranking"] = "SKIPPED_NO_SAFE_CANDIDATES"
                component_statuses["final_gate"] = "SKIPPED_NO_SAFE_CANDIDATES"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason=f"NO_SAFE_CANDIDATES: Fan-out produced 0 safe candidates. status={fan_out_res.status}, rejected={len(fan_out_res.rejected_candidates)}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="GENERATION"
                )

            # -------------------------------------------------------------
            # 8. M10 Evaluation & Ranking
            # -------------------------------------------------------------
            try:
                ranking_res: RankingResult = self.ranker.rank_fan_out(fan_out_res)
                component_statuses["ranking"] = ranking_res.status
            except Exception as exc:
                component_statuses["ranking"] = f"ERROR: {type(exc).__name__}"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason=f"RANKING_ERROR: {exc}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="GENERATION"
                )

            provenance["ranking_result"] = ranking_res.to_dict()

            if ranking_res.safe_fallback_triggered or not ranking_res.selected_candidate:
                component_statuses["final_gate"] = "SKIPPED_RANKING_FALLBACK"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason=f"RANKING_FALLBACK: {ranking_res.status}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="GENERATION"
                )

            # -------------------------------------------------------------
            # 9. M11 Final Safety & Validation Gate
            # -------------------------------------------------------------
            try:
                gate_res = self.final_gate.validate(
                    ranking_res,
                    allow_mock=self.allow_mock
                )
                component_statuses["final_gate"] = gate_res.decision
            except Exception as exc:
                component_statuses["final_gate"] = f"ERROR: {type(exc).__name__}"
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason=f"FINAL_GATE_ERROR: {exc}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="GENERATION"
                )

            provenance["final_gate_result"] = gate_res.to_dict()

            if gate_res.decision == "PASS_UNAPPROVED":
                return PipelineResult(
                    outcome="GENERATED_RESPONSE",
                    input_text=text,
                    response_text=gate_res.text,
                    candidate_id=gate_res.candidate_id,
                    source_origin="GENERATION",
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    model_id=gate_res.model_id,
                    provider=gate_res.provider,
                    quality_status=gate_res.quality_status,  # "generated_candidate"
                    review_status="unreviewed",
                    human_audited=False,
                    safe_fallback_triggered=False,
                    fallback_reason=None,
                    fallback_response=None,
                    component_statuses=component_statuses,
                    provenance=provenance,
                    disclaimer=gate_res.disclaimer
                )
            else:
                return self._create_fallback_result(
                    input_text=text,
                    is_hate=True,
                    category=cls_out.category,
                    confidence=cls_out.confidence,
                    reason=f"FINAL_GATE_REJECTED: {', '.join(gate_res.failure_reasons)}",
                    component_statuses=component_statuses,
                    provenance=provenance,
                    source_origin="GENERATION",
                    candidate_id=gate_res.candidate_id
                )

        except Exception as unhandled_exc:
            # Absolute fail-closed safety net: no uncaught exception leaves process()
            component_statuses["unhandled_exception"] = f"CRITICAL: {type(unhandled_exc).__name__}"
            return self._create_fallback_result(
                input_text=str(text) if text is not None else "",
                is_hate=False,
                category=None,
                confidence=0.0,
                reason=f"PIPELINE_CRITICAL_FAILURE: {unhandled_exc}",
                component_statuses=component_statuses,
                provenance=provenance,
                source_origin="NONE"
            )

    def _create_fallback_result(
        self,
        input_text: str,
        is_hate: bool,
        category: Optional[str],
        confidence: float,
        reason: str,
        component_statuses: Dict[str, str],
        provenance: Dict[str, Any],
        source_origin: Optional[str] = "NONE",
        candidate_id: Optional[str] = None
    ) -> PipelineResult:
        """Constructs a deterministic, auditable safe fallback result."""
        return PipelineResult(
            outcome="SAFE_FALLBACK",
            input_text=input_text,
            response_text=None,
            candidate_id=candidate_id,
            source_origin=source_origin,
            is_hate=is_hate,
            category=category,
            confidence=confidence,
            model_id=None,
            provider=None,
            quality_status="fallback",
            review_status="unreviewed",
            human_audited=False,
            safe_fallback_triggered=True,
            fallback_reason=reason,
            fallback_response=self.fallback_message,
            component_statuses=component_statuses,
            provenance=provenance,
            disclaimer=self.DISCLAIMER_FALLBACK
        )


# =====================================================================
# Factory Helpers
# =====================================================================

def create_mock_retriever(
    entries: Optional[List[Dict[str, Any]]] = None,
    dim: int = 1024
) -> CSHSMRetriever:
    """
    Constructs an in-memory CSHSMRetriever with deterministic mock embedder.
    Used for testing and offline integration.
    """
    if entries is None:
        entries = [
            {
                "kb_id": "KB-00001",
                "hate_text": "Sample religious hostility query",
                "counter_speech": "Freedom of religion is a human right. Mutual respect strengthens us all.",
                "category": "religion",
                "target": "religion",
                "source": "conan",
                "source_id": "MOCK-001",
                "quality_status": "filtered_candidate",
                "review_status": "unreviewed",
                "scope": "development_provisional",
                "row_idx": 0
            },
            {
                "kb_id": "KB-00002",
                "hate_text": "Sample gender harassment query",
                "counter_speech": "Everyone deserves dignity and safety regardless of gender.",
                "category": "gender",
                "target": "gender",
                "source": "conan",
                "source_id": "MOCK-002",
                "quality_status": "filtered_candidate",
                "review_status": "unreviewed",
                "scope": "development_provisional",
                "row_idx": 1
            },
            {
                "kb_id": "KB-00004",
                "hate_text": "You are an idiot and I hope you die.",
                "counter_speech": "You are an idiot and I hope you die.",
                "category": "religion",
                "target": "religion",
                "source": "conan",
                "source_id": "MOCK-004",
                "quality_status": "filtered_candidate",
                "review_status": "unreviewed",
                "scope": "development_provisional",
                "row_idx": 2
            }
        ]

    index = ExactInnerProductIndex(dim=dim)
    embedder = MockEmbedder(dim=dim)
    vectors = embedder.embed_batch([e["hate_text"] for e in entries])
    index.add(vectors)
    run_meta = {
        "run_id": "mock_pipeline_run",
        "scope": "development_provisional",
        "is_mock": True,
        "approval_status": "development_provisional",
        "created_at": "2026-10-24T00:00:00Z"
    }
    return CSHSMRetriever(
        index=index,
        manifest=entries,
        run_metadata=run_meta,
        embedder=embedder
    )


def create_default_pipeline(
    retriever: Optional[CSHSMRetriever] = None,
    classifier: Optional[BaseClassifier] = None,
    use_mock: bool = True,
    threshold: float = 0.80,
    fallback_message: str = PipelineOrchestrator.DEFAULT_FALLBACK_MESSAGE,
    cleaner: Optional[TextCleaner] = None,
    use_live_gemini: bool = False
) -> PipelineOrchestrator:
    """
    Factory constructing a fully wired PipelineOrchestrator.

    Modes:
      - All-Mock (default: use_mock=True, use_live_gemini=False):
          Mock classifier, mock retriever, mock fan-out (5 mocks), mock evaluator.
      - Hybrid Gemini-Live (use_live_gemini=True):
          Retains mock classifier, mock retriever, mock evaluator, and allow_mock=True,
          but activates hybrid fan-out (1 live Gemini in slot 3 + 4 mocks).
      - All-Live (use_mock=False, use_live_gemini=False):
          Production XLMR classifier, committed FAISS retriever, all-live fan-out, LLM judge evaluator.
    """
    cleaner = cleaner or TextCleaner()

    if use_live_gemini:
        if classifier is None:
            canned = {
                "Have a wonderful, peaceful, and blessed day with your family!": ClassifierOutput(
                    is_hate=False,
                    category=None,
                    confidence=0.99,
                    model_id="mock-xlmr-multitask",
                    is_mock=True,
                    metadata={"reason": "preset_non_hate"}
                ),
                "Have a wonderful and blessed day with your family!": ClassifierOutput(
                    is_hate=False,
                    category=None,
                    confidence=0.99,
                    model_id="mock-xlmr-multitask",
                    is_mock=True,
                    metadata={"reason": "preset_non_hate"}
                ),
            }
            classifier = MockClassifier(
                default_is_hate=True,
                default_category="religion",
                default_confidence=0.95,
                canned_predictions=canned
            )
        if retriever is None:
            retriever = create_mock_retriever()
        fan_out = create_default_fan_out(use_mock=True, use_live_gemini=True)
        evaluator = MockCandidateEvaluator()
        ranker = CandidateRanker(evaluator=evaluator)
        final_gate = FinalSafetyGate(fallback_message=fallback_message)
        allow_mock = True
    elif use_mock:
        if classifier is None:
            canned = {
                "Have a wonderful, peaceful, and blessed day with your family!": ClassifierOutput(
                    is_hate=False,
                    category=None,
                    confidence=0.99,
                    model_id="mock-xlmr-multitask",
                    is_mock=True,
                    metadata={"reason": "preset_non_hate"}
                ),
                "Have a wonderful and blessed day with your family!": ClassifierOutput(
                    is_hate=False,
                    category=None,
                    confidence=0.99,
                    model_id="mock-xlmr-multitask",
                    is_mock=True,
                    metadata={"reason": "preset_non_hate"}
                ),
            }
            classifier = MockClassifier(
                default_is_hate=True,
                default_category="religion",
                default_confidence=0.95,
                canned_predictions=canned
            )
        if retriever is None:
            retriever = create_mock_retriever()
        fan_out = create_default_fan_out(use_mock=True)
        evaluator = MockCandidateEvaluator()
        ranker = CandidateRanker(evaluator=evaluator)
        final_gate = FinalSafetyGate(fallback_message=fallback_message)
        allow_mock = True
    else:
        if classifier is None:
            from src.pipeline.classifier import XLMROracleClassifier
            classifier = XLMROracleClassifier()
        if retriever is None:
            retriever = CSHSMRetriever.from_committed_run(allow_mock=False)
        fan_out = create_default_fan_out(use_mock=False)
        from src.evaluation.evaluator import LLMJudgeEvaluator
        evaluator = LLMJudgeEvaluator()
        ranker = CandidateRanker(evaluator=evaluator)
        final_gate = FinalSafetyGate(fallback_message=fallback_message)
        allow_mock = False

    router = FallbackRouter(retriever=retriever, default_threshold=threshold)

    return PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=cleaner,
        allow_mock=allow_mock,
        threshold=threshold,
        fallback_message=fallback_message
    )
