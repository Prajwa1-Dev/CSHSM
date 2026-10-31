"""
Unit and integration tests for Milestone 12: Main Pipeline Wiring.
"""

import os
import sys
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.cleaner import TextCleaner
from src.retrieval.embedder import MockEmbedder
from src.retrieval.retriever import (
    CSHSMRetriever,
    ExactInnerProductIndex,
    RetrievalMatch
)
from src.routing.router import FallbackRouter, RoutingDecision
from src.generation.adapters import MockModelAdapter
from src.generation.fan_out import GenerationFanOut, create_default_fan_out
from src.evaluation.evaluator import MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker, RankingConfig
from src.safety.final_gate import FinalSafetyGate, validate_retrieved_candidate
from src.pipeline.classifier import (
    ClassifierOutput,
    BaseClassifier,
    MockClassifier,
    XLMROracleClassifier
)
from src.pipeline.orchestrator import (
    PipelineResult,
    PipelineOrchestrator,
    create_default_pipeline,
    create_mock_retriever
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_retriever():
    return create_mock_retriever()


@pytest.fixture
def standard_pipeline(mock_retriever):
    return create_default_pipeline(retriever=mock_retriever, use_mock=True, threshold=0.80)


# =====================================================================
# Tests
# =====================================================================

def test_empty_or_whitespace_input_triggers_safe_fallback(standard_pipeline):
    """Empty or whitespace input must trigger fail-closed SAFE_FALLBACK without processing."""
    res_empty = standard_pipeline.process("")
    assert res_empty.outcome == "SAFE_FALLBACK"
    assert res_empty.safe_fallback_triggered is True
    assert res_empty.fallback_response == "I cannot respond to this respectfully."
    assert "EMPTY_INPUT" in res_empty.fallback_reason
    assert res_empty.component_statuses["cleaner"] == "EMPTY_INPUT"
    assert res_empty.review_status == "unreviewed"
    assert res_empty.human_audited is False

    res_spaces = standard_pipeline.process("    \n\t  ")
    assert res_spaces.outcome == "SAFE_FALLBACK"
    assert res_spaces.safe_fallback_triggered is True

    res_none = standard_pipeline.process(None)
    assert res_none.outcome == "SAFE_FALLBACK"
    assert res_none.safe_fallback_triggered is True


def test_non_hate_early_exit(mock_retriever):
    """Non-hate input must exit early without invoking retrieval or generation."""
    cleaner = TextCleaner()
    classifier = MockClassifier(
        default_is_hate=False,
        default_category=None,
        default_confidence=0.99
    )
    router = FallbackRouter(retriever=mock_retriever, default_threshold=0.80)
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=cleaner,
        allow_mock=True
    )

    result = pipeline.process("Have a wonderful and blessed day!")
    assert result.outcome == "NON_HATE_EARLY_EXIT"
    assert result.is_hate is False
    assert result.response_text is None
    assert result.candidate_id is None
    assert result.source_origin == "NONE"
    assert result.quality_status == "non_hate_exit"
    assert result.review_status == "unreviewed"
    assert result.human_audited is False
    assert result.safe_fallback_triggered is False
    assert result.fallback_response is None
    assert result.component_statuses["router"] == "SKIPPED"
    assert result.component_statuses["generation"] == "SKIPPED"
    assert result.component_statuses["ranking"] == "SKIPPED"
    assert result.component_statuses["final_gate"] == "SKIPPED"


def test_hate_input_high_similarity_routes_to_retrieved_response(mock_retriever):
    """Hate input with similarity >= threshold routes to retrieved candidate through M11 gate."""
    # The mock retriever has "Sample religious hostility query"
    cleaner = TextCleaner()
    classifier = MockClassifier(
        default_is_hate=True,
        default_category="religion",
        default_confidence=0.96
    )
    # Set threshold low enough so identical or similar query matches
    router = FallbackRouter(retriever=mock_retriever, default_threshold=0.50)
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=cleaner,
        allow_mock=True,
        threshold=0.50
    )

    query = "Sample religious hostility query"
    result = pipeline.process(query)

    assert result.outcome == "RETRIEVED_RESPONSE"
    assert result.is_hate is True
    assert result.source_origin == "RETRIEVAL"
    assert result.candidate_id == "KB-00001"
    assert result.response_text == "Freedom of religion is a human right. Mutual respect strengthens us all."
    assert result.quality_status == "filtered_candidate"  # Never falsely marked as generated or approved
    assert result.review_status == "unreviewed"
    assert result.human_audited is False
    assert result.safe_fallback_triggered is False
    assert result.component_statuses["generation"] == "SKIPPED"
    assert result.component_statuses["ranking"] == "SKIPPED"
    assert result.component_statuses["final_gate"] == "PASS_UNAPPROVED"
    assert "VALIDATED UNAPPROVED RETRIEVED CANDIDATE" in result.disclaimer


def test_hate_input_low_similarity_routes_to_generated_response(mock_retriever):
    """Hate input with similarity < threshold routes to M9 -> M10 -> M11 generated response."""
    cleaner = TextCleaner()
    classifier = MockClassifier(
        default_is_hate=True,
        default_category="caste",
        default_confidence=0.92
    )
    # Threshold 1.0001 guarantees low similarity fallback
    router = FallbackRouter(retriever=mock_retriever, default_threshold=1.0001)
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=cleaner,
        allow_mock=True,
        threshold=1.0001
    )

    query = "Completely novel and low similarity caste hate comment"
    result = pipeline.process(query)

    assert result.outcome == "GENERATED_RESPONSE"
    assert result.is_hate is True
    assert result.source_origin == "GENERATION"
    assert result.response_text is not None
    assert len(result.response_text) > 10
    assert result.candidate_id is not None
    assert result.quality_status == "generated_candidate"
    assert result.review_status == "unreviewed"
    assert result.human_audited is False
    assert result.safe_fallback_triggered is False
    assert result.component_statuses["router"] == "FALLBACK_GENERATION"
    assert result.component_statuses["generation"] == "SUCCESS"
    assert result.component_statuses["ranking"] in ("SELECTED", "SUCCESS")
    assert result.component_statuses["final_gate"] == "PASS_UNAPPROVED"


def test_m11_safety_filter_rejects_unsafe_retrieved_candidate():
    """Defense-in-depth: Unsafe retrieved candidate is rejected by M11 and triggers safe fallback."""
    # Candidate with slur/hate text in counter_speech
    entries = [
        {
            "kb_id": "KB-BAD-01",
            "hate_text": "Sample hate speech",
            "counter_speech": "You should kill yourself and die.",  # Prohibited toxic phrase
            "category": "religion",
            "target": "religion",
            "source": "conan",
            "source_id": "MOCK-BAD",
            "quality_status": "filtered_candidate",
            "review_status": "unreviewed",
            "scope": "development_provisional",
            "row_idx": 0
        }
    ]
    bad_retriever = create_mock_retriever(entries=entries)
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    router = FallbackRouter(retriever=bad_retriever, default_threshold=0.0)  # Always retrieve
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=True,
        threshold=0.0
    )

    result = pipeline.process("Sample hate speech")
    assert result.outcome == "SAFE_FALLBACK"
    assert result.safe_fallback_triggered is True
    assert result.source_origin == "RETRIEVAL"
    assert result.response_text is None
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert "RETRIEVED_SAFETY_REJECTED" in result.fallback_reason
    assert result.component_statuses["final_gate"] == "REJECTED"


def test_m11_safety_filter_rejects_unsafe_generated_candidate(mock_retriever):
    """M11 final gate and upstream safety filter reject toxic generated content."""
    unsafe_response = "You are an idiot and I hope you die."
    adapters = [
        MockModelAdapter(model_id="mock-1", provider="mock", canned_response=unsafe_response)
    ]
    fan_out = GenerationFanOut(adapters=adapters)
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    router = FallbackRouter(retriever=mock_retriever, default_threshold=1.0001)  # Force generation
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=True,
        threshold=1.0001
    )

    result = pipeline.process("Hate text")
    assert result.outcome == "SAFE_FALLBACK"
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert ("NO_SAFE_CANDIDATES" in result.fallback_reason or
            "FINAL_GATE_REJECTED" in result.fallback_reason or
            "RANKING_FALLBACK" in result.fallback_reason)


def test_m11_final_gate_rejects_candidate_below_gate_threshold(mock_retriever):
    """M11 final gate enforces defense-in-depth gate thresholds independently."""
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    router = FallbackRouter(retriever=mock_retriever, default_threshold=1.0001)
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator(default_scores={"safety": 4, "relevance": 4, "de_escalation": 4, "persuasiveness": 4}))
    # Set final gate min_safety_score to 5 (higher than candidate's 4)
    final_gate = FinalSafetyGate(min_safety_score=5)

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=True,
        threshold=1.0001
    )

    result = pipeline.process("Hate text")
    assert result.outcome == "SAFE_FALLBACK"
    assert result.safe_fallback_triggered is True
    assert "FINAL_GATE_REJECTED" in result.fallback_reason
    assert result.component_statuses["final_gate"] == "REJECTED"


def test_all_candidates_ineligible_in_m10_triggers_safe_fallback(mock_retriever):
    """When all candidates fail M10 hard gates, pipeline triggers safe fallback."""
    # Evaluator returning safety score = 1 (failing mandatory gate min_safety_score=4)
    low_safety_evaluator = MockCandidateEvaluator(default_scores={"safety": 1, "relevance": 5, "de_escalation": 5, "persuasiveness": 5})
    ranker = CandidateRanker(evaluator=low_safety_evaluator)
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    router = FallbackRouter(retriever=mock_retriever, default_threshold=1.0001)
    fan_out = create_default_fan_out(use_mock=True)
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=True,
        threshold=1.0001
    )

    result = pipeline.process("Hate speech query")
    assert result.outcome == "SAFE_FALLBACK"
    assert result.safe_fallback_triggered is True
    assert "RANKING_FALLBACK" in result.fallback_reason
    assert result.fallback_response == "I cannot respond to this respectfully."


def test_router_error_gracefully_falls_back_to_generation(mock_retriever):
    """If retriever raises an unexpected error, pipeline safely routes to generation."""
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    router = FallbackRouter(retriever=mock_retriever, default_threshold=0.80)

    # Monkeypatch route_text to raise an unexpected runtime error
    def crashing_route_text(*args, **kwargs):
        raise RuntimeError("Vector database network partition simulated.")

    router.route_text = crashing_route_text

    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=True
    )

    result = pipeline.process("Some hate text")
    assert result.outcome == "GENERATED_RESPONSE"
    assert result.source_origin == "GENERATION"
    assert result.safe_fallback_triggered is False
    assert result.response_text is not None


def test_classifier_error_fails_closed(mock_retriever):
    """Classifier failure must fail closed to safe fallback, never assuming non-hate."""
    classifier = MockClassifier(
        exception_to_raise=RuntimeError("XLM-R GPU Out of Memory")
    )
    router = FallbackRouter(retriever=mock_retriever, default_threshold=0.80)
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=True
    )

    result = pipeline.process("Hate text")
    assert result.outcome == "SAFE_FALLBACK"
    assert result.safe_fallback_triggered is True
    assert "CLASSIFIER_ERROR" in result.fallback_reason
    assert result.component_statuses["classifier"].startswith("ERROR")
    assert result.fallback_response == "I cannot respond to this respectfully."


def test_xlmr_oracle_classifier_fails_closed_when_dependencies_missing():
    """XLMROracleClassifier must fail closed when dependencies or weights are missing."""
    with pytest.raises((RuntimeError, FileNotFoundError)) as exc_info:
        XLMROracleClassifier(model_path="nonexistent_checkpoint.pt")
    assert ("Required library is missing" in str(exc_info.value) or
            "checkpoint not found" in str(exc_info.value))


def test_allow_mock_false_rejects_mock_evidence(mock_retriever):
    """When allow_mock=False, mock classifier or components must be rejected fail-closed."""
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    router = FallbackRouter(retriever=mock_retriever, default_threshold=0.80)
    fan_out = create_default_fan_out(use_mock=True)
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    final_gate = FinalSafetyGate()

    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=TextCleaner(),
        allow_mock=False  # Production mode
    )

    result = pipeline.process("Sample hate text")
    assert result.outcome == "SAFE_FALLBACK"
    assert result.safe_fallback_triggered is True
    assert "MOCK_PROHIBITED" in result.fallback_reason
    assert result.component_statuses["classifier"] == "MOCK_PROHIBITED"


def test_pipeline_result_serialization_and_immutability(standard_pipeline):
    """PipelineResult must serialize cleanly to JSON and strictly uphold invariants."""
    result = standard_pipeline.process("You belong to an inferior religious group.")
    data = result.to_dict()

    # JSON round-trip
    serialized = json.dumps(data)
    loaded = json.loads(serialized)

    assert loaded["outcome"] in ("RETRIEVED_RESPONSE", "GENERATED_RESPONSE", "SAFE_FALLBACK")
    assert loaded["human_audited"] is False
    assert loaded["review_status"] == "unreviewed"
    assert loaded["quality_status"] in ("filtered_candidate", "generated_candidate", "fallback")
    assert "component_statuses" in loaded
    assert "provenance" in loaded
    assert "disclaimer" in loaded


# =====================================================================
# Hybrid Gemini-Live Pipeline Tests
# =====================================================================

def test_hybrid_pipeline_keeps_mock_components_and_allow_mock_true(mock_retriever):
    pipeline = create_default_pipeline(retriever=mock_retriever, use_live_gemini=True)

    assert pipeline.allow_mock is True
    assert isinstance(pipeline.classifier, MockClassifier)
    assert isinstance(pipeline.ranker.evaluator, MockCandidateEvaluator)
    assert pipeline.router.retriever.embedder.is_mock is True

    # Fan-out: exactly slot 3 is real Google, others mock
    adapters = pipeline.fan_out.adapters
    assert len(adapters) == 5
    assert adapters[2].is_mock is False
    assert adapters[2].provider == "google"
    assert all(adapters[i].is_mock for i in (0, 1, 3, 4))


def test_hybrid_pipeline_with_unconfigured_gemini_fails_closed_safely(mock_retriever, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    pipeline = create_default_pipeline(retriever=mock_retriever, use_live_gemini=True, threshold=0.99)
    # High threshold forces fallback to generation fan-out
    result = pipeline.process("Hateful query that routes to generation")

    assert result.outcome == "GENERATED_RESPONSE"
    assert result.safe_fallback_triggered is False
    assert result.candidate_id is not None
    # Provenance tracks generation fan-out with partial success
    assert "fan_out_result" in result.provenance
    assert result.provenance["fan_out_result"]["status"] == "PARTIAL_SUCCESS"
