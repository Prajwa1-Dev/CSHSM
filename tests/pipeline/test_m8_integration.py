"""
Automated unit and integration tests for CSHSM Milestone 8:
Finalized 5-Model Fan-Out, Dedicated Gemini Judge, KB V2 Retrieval, and 8 Operational Scenarios.

Verifies:
1. Factory construction defaults to completely offline mock mode (use_mock=True).
2. Explicit KB V2 verification fails closed if artifacts are missing (anti-fallback guard).
3. Backend mode selection (CSHSM_PIPELINE_MODE=m8) defaults to offline mock execution.
4. Backend mode selection fails clearly if conflicting live and mock flags are provided.
5. Backend preserves existing default and legacy Phase 3 behavior.
6. Scenario 1: Non-hate early exit (zero generator calls, zero retrieval calls).
7. Scenario 2: Successful KB V2 retrieval (similarity >= 0.80, zero generator calls).
8. Scenario 3: Five-model generation fan-out fallback (similarity < 0.80).
9. Scenario 4: Partial provider failure resiliency (surviving valid candidates proceed).
10. Scenario 5: Gemini evaluation failure fail-closed behavior (safe fallback).
11. Scenario 6: All candidates rejected by hard gates (safe fallback).
12. Scenario 7: Final safety gate defense-in-depth rejection (safe fallback).
13. Scenario 8: Empty and whitespace input safe handling (safe fallback).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.demonstrate_m8_pipeline import (
    build_m8_pipeline,
    _verify_kb_v2_artifacts,
    run_scenarios,
    get_demonstration_scenarios
)
from backend.app.main import create_app
from src.pipeline.classifier import MockClassifier, ClassifierOutput
from src.retrieval.retriever import CSHSMRetriever
from src.generation.adapters import MockModelAdapter
from src.evaluation.evaluator import MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker
from src.safety.final_gate import FinalSafetyGate
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineResult


# =====================================================================
# 1. Factory Construction & Safety Guards
# =====================================================================

def test_build_m8_pipeline_default_mock():
    """Verifies that build_m8_pipeline defaults to completely offline mock execution."""
    pipeline = build_m8_pipeline(use_mock=True)

    assert isinstance(pipeline, PipelineOrchestrator)
    assert pipeline.allow_mock is True
    assert isinstance(pipeline.classifier, MockClassifier)
    assert isinstance(pipeline.final_gate, FinalSafetyGate)
    assert isinstance(pipeline.ranker.evaluator, MockCandidateEvaluator)

    # Verify 5 mock generator models matching finalized shortlist
    adapters = pipeline.fan_out.adapters
    assert len(adapters) == 5
    model_ids = [a.model_id for a in adapters]
    assert model_ids == [
        "mock-gemma4-26b-it",
        "mock-glm-4.7-flash",
        "mock-gpt-oss-20b",
        "mock-qwen-3.8-27b",
        "mock-gpt-oss-120b"
    ]
    for a in adapters:
        assert isinstance(a, MockModelAdapter)
        assert a.is_mock is True

    # Verify timeout and threshold
    assert pipeline.fan_out.overall_timeout == 70.0
    assert pipeline.threshold == 0.80


def test_kb_v2_explicit_verification_fails_closed():
    """Verifies that _verify_kb_v2_artifacts fails closed when artifacts are missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        _verify_kb_v2_artifacts("non_existent_kb_v2_dir_12345")
    assert "CSHSM M8 requires KB V2 embeddings directory" in str(exc_info.value)
    assert "Silent fallback to demo KB is strictly prohibited" in str(exc_info.value)

    # Verifies that authoritative committed run passes verification
    _verify_kb_v2_artifacts("data/counter_speech_kb/embeddings_v2")


# =====================================================================
# 2. Backend Mode Selection & Configuration Guards
# =====================================================================

def test_backend_m8_mode_defaults_to_offline_mock(monkeypatch):
    """Verifies that CSHSM_PIPELINE_MODE=m8 defaults to offline mock mode without live opt-in."""
    monkeypatch.setenv("CSHSM_PIPELINE_MODE", "m8")
    monkeypatch.delenv("CSHSM_M8_LIVE", raising=False)
    monkeypatch.delenv("CSHSM_M8_MOCK", raising=False)

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True


def test_backend_m8_mode_conflicting_flags_raise_error(monkeypatch):
    """Verifies that providing both CSHSM_M8_LIVE and CSHSM_M8_MOCK raises a clear ValueError."""
    monkeypatch.setenv("CSHSM_PIPELINE_MODE", "m8")
    monkeypatch.setenv("CSHSM_M8_LIVE", "1")
    monkeypatch.setenv("CSHSM_M8_MOCK", "1")

    with pytest.raises(ValueError) as exc_info:
        create_app(test_config={"TESTING": True})
    assert "Conflicting CSHSM M8 configuration" in str(exc_info.value)


def test_backend_preserves_phase3_demo_behavior(monkeypatch):
    """Verifies that setting CSHSM_DEMO_PIPELINE=1 preserves legacy Phase 3 demonstration behavior."""
    monkeypatch.delenv("CSHSM_PIPELINE_MODE", raising=False)
    monkeypatch.setenv("CSHSM_DEMO_PIPELINE", "1")

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_backend_preserves_default_mock_mode(monkeypatch):
    """Verifies that default create_app without environment variables remains offline mock."""
    monkeypatch.delenv("CSHSM_PIPELINE_MODE", raising=False)
    monkeypatch.delenv("CSHSM_DEMO_PIPELINE", raising=False)
    monkeypatch.delenv("CSHSM_USE_LIVE_GEMINI", raising=False)

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mock_mode"] is True


# =====================================================================
# 3. Operational Scenarios 1 through 8 Verification
# =====================================================================

@pytest.fixture
def m8_pipeline():
    """Provides a fully wired mock M8 pipeline."""
    return build_m8_pipeline(use_mock=True)


def test_scenario_1_non_hate_early_exit(m8_pipeline):
    """Scenario 1: Non-hate input short-circuits with 0 generator calls and 0 retrieval calls."""
    text = "Wishing everyone a peaceful and productive week ahead! Let's collaborate together."
    res = m8_pipeline.process(text)

    assert res.outcome == "NON_HATE_EARLY_EXIT"
    assert res.is_hate is False
    assert res.category is None
    assert res.safe_fallback_triggered is False
    assert res.component_statuses["cleaner"] == "SUCCESS"
    assert res.component_statuses["classifier"] == "SUCCESS"
    assert res.component_statuses["router"] == "SKIPPED"
    assert res.component_statuses["generation"] == "SKIPPED"
    assert res.component_statuses["ranking"] == "SKIPPED"
    assert res.component_statuses["final_gate"] == "SKIPPED"


def test_scenario_2_high_similarity_retrieval(m8_pipeline):
    """Scenario 2: High-similarity hate input routes directly to retrieved candidate."""
    text = "Hudugir ge driving baralla, road alli bandre accident guarantee."
    res = m8_pipeline.process(text)

    assert res.outcome == "RETRIEVED_RESPONSE"
    assert res.is_hate is True
    assert res.category.lower() == "gender"
    assert res.source_origin == "RETRIEVAL"
    assert res.candidate_id in ("CSHSM-KN-00001", "KB-V2-00001")
    assert "Driving skill" in res.response_text
    assert res.safe_fallback_triggered is False
    assert res.component_statuses["router"] == "RETRIEVE"
    assert res.component_statuses["generation"] == "SKIPPED"
    assert res.component_statuses["ranking"] == "SKIPPED"
    assert res.component_statuses["final_gate"] == "PASS_UNAPPROVED"


def test_scenario_3_low_similarity_fan_out(m8_pipeline):
    """Scenario 3: Novel hate input routes to 5-model fan-out, evaluation, and ranking."""
    text = "Every single member of that community is a criminal and should be deported immediately."
    res = m8_pipeline.process(text)

    assert res.outcome == "GENERATED_RESPONSE"
    assert res.is_hate is True
    assert res.source_origin == "GENERATION"
    assert res.safe_fallback_triggered is False
    assert res.component_statuses["router"] == "FALLBACK_GENERATION"
    assert res.component_statuses["generation"] == "SUCCESS"
    assert res.component_statuses["ranking"] == "SELECTED"
    assert res.component_statuses["final_gate"] == "PASS_UNAPPROVED"


def test_scenario_4_partial_provider_failures(m8_pipeline):
    """Scenario 4: Partial provider failures still collect surviving candidates and succeed."""
    scenarios = run_scenarios(orchestrator=m8_pipeline, use_mock=True, scenario_id_filter=4)
    assert len(scenarios) == 1
    sc = scenarios[0]

    assert sc["outcome"] == "GENERATED_RESPONSE"
    assert sc["safe_fallback_triggered"] is False
    assert sc["component_statuses"]["generation"] == "PARTIAL_SUCCESS"
    assert sc["component_statuses"]["ranking"] == "SELECTED"
    assert sc["component_statuses"]["final_gate"] == "PASS_UNAPPROVED"


def test_scenario_5_gemini_evaluation_failure_fails_closed(m8_pipeline):
    """Scenario 5: Gemini evaluation failure triggers fail-closed safe fallback."""
    scenarios = run_scenarios(orchestrator=m8_pipeline, use_mock=True, scenario_id_filter=5)
    assert len(scenarios) == 1
    sc = scenarios[0]

    assert sc["outcome"] == "SAFE_FALLBACK"
    assert sc["safe_fallback_triggered"] is True
    assert "RANKING_FALLBACK: EVALUATION_FAILED" in sc["fallback_reason"]
    assert sc["component_statuses"]["ranking"] == "EVALUATION_FAILED"
    assert sc["fallback_response"] == "I cannot respond to this respectfully."


def test_scenario_6_all_candidates_rejected_by_hard_gates(m8_pipeline):
    """Scenario 6: All candidate scores failing hard gates triggers safe fallback."""
    scenarios = run_scenarios(orchestrator=m8_pipeline, use_mock=True, scenario_id_filter=6)
    assert len(scenarios) == 1
    sc = scenarios[0]

    assert sc["outcome"] == "SAFE_FALLBACK"
    assert sc["safe_fallback_triggered"] is True
    assert "RANKING_FALLBACK: ALL_INELIGIBLE" in sc["fallback_reason"]
    assert sc["component_statuses"]["ranking"] == "ALL_INELIGIBLE"
    assert sc["fallback_response"] == "I cannot respond to this respectfully."


def test_scenario_7_final_safety_gate_rejection(m8_pipeline):
    """Scenario 7: Final safety gate defense-in-depth rejects candidate and triggers fallback."""
    scenarios = run_scenarios(orchestrator=m8_pipeline, use_mock=True, scenario_id_filter=7)
    assert len(scenarios) == 1
    sc = scenarios[0]

    assert sc["outcome"] == "SAFE_FALLBACK"
    assert sc["safe_fallback_triggered"] is True
    assert "FINAL_GATE_REJECTED" in sc["fallback_reason"]
    assert sc["component_statuses"]["final_gate"] == "REJECTED"
    assert sc["fallback_response"] == "I cannot respond to this respectfully."


def test_scenario_8_empty_whitespace_input_safe_handling(m8_pipeline):
    """Scenario 8: Empty or whitespace input is caught by input validator and safely triggers fallback."""
    text = "   \t\n  "
    res = m8_pipeline.process(text)

    assert res.outcome == "SAFE_FALLBACK"
    assert res.safe_fallback_triggered is True
    assert res.component_statuses["cleaner"] == "EMPTY_INPUT"
    assert res.component_statuses["classifier"] == "PENDING"
    assert res.component_statuses["router"] == "PENDING"
    assert res.component_statuses["generation"] == "PENDING"
    assert res.fallback_response == "I cannot respond to this respectfully."


# =====================================================================
# 4. Six-Category Full-Pipeline Propagation Tests
# =====================================================================

@pytest.mark.parametrize("category", [
    "Gender",
    "Political",
    "Religion",
    "Geo-political",
    "Violence",
    "Others"
])
def test_all_six_categories_pipeline_propagation(category):
    """
    Verifies that all six taxonomical categories propagate correctly through
    the complete PipelineOrchestrator:
      1. Classifier category reaches the router.
      2. Router propagates category into generation fan-out when fallback is triggered.
      3. Final PipelineResult preserves the correct category.
      4. Zero live provider calls occur.
    """
    query = f"Hate statement targeting {category} collective identity."
    canned = {
        query: ClassifierOutput(
            is_hate=True,
            category=category,
            confidence=0.96,
            model_id="mock-xlmr-multitask",
            is_mock=True
        )
    }
    classifier = MockClassifier(
        default_is_hate=True,
        default_category=category,
        canned_predictions=canned
    )

    pipeline = build_m8_pipeline(use_mock=True, classifier=classifier)
    res = pipeline.process(query)

    # Invariants
    assert res.is_hate is True
    assert res.category == category
    assert res.outcome == "GENERATED_RESPONSE"
    assert res.safe_fallback_triggered is False
    assert res.component_statuses["cleaner"] == "SUCCESS"
    assert res.component_statuses["classifier"] == "SUCCESS"
    assert res.component_statuses["router"] == "FALLBACK_GENERATION"
    assert res.component_statuses["generation"] == "SUCCESS"
    assert res.component_statuses["ranking"] == "SELECTED"
    assert res.component_statuses["final_gate"] == "PASS_UNAPPROVED"

    # Verify category reached GenerationRequest in fan-out
    fan_out_res = res.provenance.get("fan_out_result", {})
    req = fan_out_res.get("request", {})
    assert req.get("category") == category


# =====================================================================
# 5. Hybrid Offline Real KB V2 Integration Tests
# =====================================================================

def test_m8_offline_hybrid_pipeline_real_kb_v2_integration():
    """
    Verifies that the offline M8 dashboard pipeline integrates the real 300-record KB V2,
    retrieves canonical Kanglish counter-speech CSHSM-KN-00001 for high similarity,
    falls back to 5 mock models for novel input, and remains completely offline.
    """
    pipeline = build_m8_pipeline(use_mock=True, use_real_kb=True)

    # 1. Verify 300 records loaded in FAISS backend
    retriever = pipeline.router.retriever
    assert retriever.index.ntotal == 300
    assert retriever.index.backend_name == "faiss"
    assert retriever.run_metadata["is_mock"] is False

    # 2. Test exact Kanglish input retrieves CSHSM-KN-00001
    query = "Hudugir ge driving baralla, road alli bandre accident guarantee."
    res = pipeline.process(query)

    assert res.outcome == "RETRIEVED_RESPONSE"
    assert res.is_hate is True
    assert res.category in ("Gender", "gender")
    assert res.candidate_id == "CSHSM-KN-00001"
    assert res.source_origin == "RETRIEVAL"
    assert "Driving skill gender mele depend aagalla" in res.response_text
    assert res.component_statuses["router"] == "RETRIEVE"
    assert res.component_statuses["generation"] == "SKIPPED"
    assert res.component_statuses["final_gate"] == "PASS_UNAPPROVED"

    # 3. Test novel input (< 0.80) routes to mock generation fan-out
    novel_query = "Every single member of that community is a criminal and should be deported immediately."
    res_novel = pipeline.process(novel_query)

    assert res_novel.outcome == "GENERATED_RESPONSE"
    assert res_novel.source_origin == "GENERATION"
    assert res_novel.component_statuses["router"] == "FALLBACK_GENERATION"
    assert res_novel.component_statuses["generation"] == "SUCCESS"
    assert res_novel.component_statuses["ranking"] == "SELECTED"
    assert res_novel.component_statuses["final_gate"] == "PASS_UNAPPROVED"

    # 4. Verify mock adapters and evaluator remain mocked
    for adapter in pipeline.fan_out.adapters:
        assert adapter.is_mock is True
    assert pipeline.ranker.evaluator.is_mock is True


def test_backend_m8_mode_api_endpoint_retrieves_real_kb_v2(monkeypatch):
    """
    Verifies that Flask /api/process endpoint under CSHSM_PIPELINE_MODE=m8
    retrieves canonical Kanglish counter-speech from real KB V2 through the HTTP REST API.
    """
    monkeypatch.setenv("CSHSM_PIPELINE_MODE", "m8")
    monkeypatch.delenv("CSHSM_M8_LIVE", raising=False)
    monkeypatch.delenv("CSHSM_M8_MOCK", raising=False)

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()

    resp = client.post("/api/process", json={"text": "Hudugir ge driving baralla, road alli bandre accident guarantee."})
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["outcome"] == "RETRIEVED_RESPONSE"
    assert data["candidate_id"] == "CSHSM-KN-00001"
    assert data["source_origin"] == "RETRIEVAL"
    assert "Driving skill gender mele depend aagalla" in data["response_text"]
    assert data["component_statuses"]["router"] == "RETRIEVE"
    assert data["component_statuses"]["generation"] == "SKIPPED"


def test_build_m8_pipeline_real_classifier_offline_hybrid():
    """
    Verifies that build_m8_pipeline with use_real_classifier=True:
      - Uses real XLMROracleClassifier.
      - Uses real BGE-M3 + 300-record FAISS KB V2.
      - Keeps all 5 generation adapters and evaluator mocked.
      - Executes completely offline with zero external provider calls.
    """
    pipeline = build_m8_pipeline(use_mock=True, use_real_classifier=True)
    assert type(pipeline.classifier).__name__ == "XLMROracleClassifier"
    assert getattr(pipeline.classifier, "is_mock", False) is False
    assert pipeline.allow_mock is True
    assert pipeline.router.retriever.index.ntotal == 300
    assert pipeline.router.retriever.index.backend_name == "faiss"
    for adapter in pipeline.fan_out.adapters:
        assert adapter.is_mock is True
    assert pipeline.ranker.evaluator.is_mock is True

    # Test non-hate early exit on benign civic input
    res_benign = pipeline.process("Namma city clean agi irbeku andre ellaru responsible agi behave madbeku.")
    assert res_benign.outcome == "NON_HATE_EARLY_EXIT"
    assert res_benign.is_hate is False
    assert res_benign.component_statuses["classifier"] == "SUCCESS"
    assert res_benign.component_statuses["router"] == "SKIPPED"


def test_backend_m8_mode_with_real_classifier(monkeypatch):
    """
    Verifies Flask REST API under CSHSM_PIPELINE_MODE=m8 and CSHSM_M8_USE_REAL_CLASSIFIER=1:
      - /api/health exposes 'classifier_type': 'XLMROracleClassifier'.
      - /api/process executes real XLM-R inference with mock generation/judge offline.
    """
    monkeypatch.setenv("CSHSM_PIPELINE_MODE", "m8")
    monkeypatch.setenv("CSHSM_M8_USE_REAL_CLASSIFIER", "1")
    monkeypatch.delenv("CSHSM_M8_LIVE", raising=False)
    monkeypatch.delenv("CSHSM_M8_MOCK", raising=False)

    app = create_app(test_config={"TESTING": True})
    client = app.test_client()

    # 1. Health check
    resp_h = client.get("/api/health")
    assert resp_h.status_code == 200
    data_h = resp_h.get_json()
    assert data_h["classifier_type"] == "XLMROracleClassifier"
    assert data_h["mock_mode"] is True

    # 2. Benign input early exit
    resp_p = client.post("/api/process", json={"text": "Namma city clean agi irbeku andre ellaru responsible agi behave madbeku."})
    assert resp_p.status_code == 200
    data_p = resp_p.get_json()
    assert data_p["outcome"] == "NON_HATE_EARLY_EXIT"
    assert data_p["is_hate"] is False

