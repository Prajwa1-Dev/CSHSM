"""
Unit and integration tests for Milestone 13: Flask REST API Backend.

Verifies:
1. Liveness and diagnostic health checks (/health and /api/health).
2. CORS headers across GET, POST, and OPTIONS requests.
3. Input validation and 400 Bad Request handling for malformed or missing payloads.
4. Non-hate early exit flow through /api/process (NON_HATE_EARLY_EXIT).
5. High-similarity retrieval flow through /api/process (RETRIEVED_RESPONSE).
6. Low-similarity generation fan-out flow through /api/process (GENERATED_RESPONSE).
7. Safe fallback flow on empty/whitespace input (SAFE_FALLBACK).
8. Knowledge base statistics (/api/kb/stats) reporting unapproved status.
9. Review queue endpoint (/api/kb/review-queue) returning candidates and AI assessments.
"""

import os
import sys
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.main import create_app
from src.pipeline.orchestrator import (
    create_default_pipeline,
    create_mock_retriever,
    PipelineOrchestrator
)
from src.pipeline.classifier import MockClassifier
from src.routing.router import FallbackRouter
from src.generation.fan_out import create_default_fan_out
from src.evaluation.evaluator import MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker
from src.safety.final_gate import FinalSafetyGate
from src.data.cleaner import TextCleaner


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_retriever():
    return create_mock_retriever()


@pytest.fixture
def client(mock_retriever):
    pipeline = create_default_pipeline(retriever=mock_retriever, use_mock=True, threshold=0.80)
    app = create_app(orchestrator=pipeline, test_config={"TESTING": True})
    return app.test_client()


# =====================================================================
# Tests: Health, Diagnostics & CORS
# =====================================================================

def test_health_endpoints(client):
    """GET /health and GET /api/health must return 200 OK with diagnostics."""
    res1 = client.get("/health")
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert data1["status"] == "ok"
    assert data1["service"] == "CSHSM REST API"
    assert data1["framework"] == "Flask 3.1.3"
    assert data1["mock_mode"] is True

    res2 = client.get("/api/health")
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["status"] == "ok"
    assert "timestamp" in data2
    assert "threshold" in data2


def test_cors_headers_applied(client):
    """CORS headers must be attached to responses and OPTIONS preflight must return 204."""
    # Preflight
    res_opts = client.options("/api/process")
    assert res_opts.status_code == 204
    assert res_opts.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in res_opts.headers.get("Access-Control-Allow-Methods", "")

    # Normal GET
    res_get = client.get("/api/health")
    assert res_get.headers.get("Access-Control-Allow-Origin") == "*"


# =====================================================================
# Tests: Research Dashboard UI (Milestone 14)
# =====================================================================

def test_dashboard_serves_html(client):
    """GET / must serve the self-contained research dashboard with 200 OK."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.content_type.startswith("text/html")

    html = res.get_data(as_text=True)
    # Header & Badges
    assert "Counter-Speech Research System" in html
    assert "MOCK RESEARCH PROTOTYPE" in html
    assert "Flask 3.1.3" in html

    # Input & Controls
    assert "comment-input" in html
    assert "btn-analyze" in html
    assert "setPreset('non_hate')" in html
    assert "setPreset('retrieval')" in html
    assert "setPreset('generation')" in html

    # Pipeline DAG Nodes
    assert "dag-cleaner" in html
    assert "dag-classifier" in html
    assert "dag-router" in html
    assert "dag-generation" in html
    assert "dag-ranking" in html
    assert "dag-final_gate" in html

    # Outcome Panel & Telemetry
    assert "outcome-banner" in html
    assert "outcome-badge" in html
    assert "response-box" in html
    assert "meta-pills" in html
    assert "json-telemetry" in html
    assert "queue-table-body" in html
    assert "tab-kb" in html


# =====================================================================
# Tests: Input Validation & Bad Request Handling
# =====================================================================

def test_process_invalid_json_returns_400(client):
    """Malformed or non-JSON payloads must be rejected with 400 Bad Request."""
    res = client.post(
        "/api/process",
        data="not-a-json-string",
        content_type="application/json"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["error"] == "Bad Request"


def test_process_missing_text_field_returns_400(client):
    """Payloads missing the 'text' key must be rejected with 400 Bad Request."""
    res = client.post("/api/process", json={"query": "missing text key"})
    assert res.status_code == 400
    data = res.get_json()
    assert "required" in data["message"]


def test_process_non_string_text_returns_400(client):
    """Payloads where 'text' is not a string must be rejected with 400 Bad Request."""
    res = client.post("/api/process", json={"text": 12345})
    assert res.status_code == 400
    data = res.get_json()
    assert "must be a string" in data["message"]


# =====================================================================
# Tests: Pipeline Routing via /api/process
# =====================================================================

def test_process_non_hate_early_exit(mock_retriever):
    """Non-hate input triggers early exit with response_text=None and 200 OK."""
    classifier = MockClassifier(default_is_hate=False, default_confidence=0.98)
    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=FallbackRouter(retriever=mock_retriever, default_threshold=0.80),
        fan_out=create_default_fan_out(use_mock=True),
        ranker=CandidateRanker(evaluator=MockCandidateEvaluator()),
        final_gate=FinalSafetyGate(),
        cleaner=TextCleaner(),
        allow_mock=True
    )
    app = create_app(orchestrator=pipeline, test_config={"TESTING": True})
    client = app.test_client()

    res = client.post("/api/process", json={"text": "Have a wonderful, peaceful day."})
    assert res.status_code == 200
    data = res.get_json()

    assert data["outcome"] == "NON_HATE_EARLY_EXIT"
    assert data["is_hate"] is False
    assert data["response_text"] is None
    assert data["source_origin"] == "NONE"
    assert data["safe_fallback_triggered"] is False
    assert data["component_statuses"]["router"] == "SKIPPED"
    assert data["component_statuses"]["generation"] == "SKIPPED"


def test_process_retrieval_match(mock_retriever):
    """Hate input meeting threshold routes to retrieved candidate through M11 gate."""
    classifier = MockClassifier(default_is_hate=True, default_category="religion")
    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=FallbackRouter(retriever=mock_retriever, default_threshold=0.50),
        fan_out=create_default_fan_out(use_mock=True),
        ranker=CandidateRanker(evaluator=MockCandidateEvaluator()),
        final_gate=FinalSafetyGate(),
        cleaner=TextCleaner(),
        allow_mock=True,
        threshold=0.50
    )
    app = create_app(orchestrator=pipeline, test_config={"TESTING": True})
    client = app.test_client()

    # Query matches mock_retriever's "Sample religious hostility query"
    res = client.post("/api/process", json={"text": "Sample religious hostility query"})
    assert res.status_code == 200
    data = res.get_json()

    assert data["outcome"] == "RETRIEVED_RESPONSE"
    assert data["is_hate"] is True
    assert data["source_origin"] == "RETRIEVAL"
    assert data["candidate_id"] == "KB-00001"
    assert "Freedom of religion is a human right" in data["response_text"]
    assert data["quality_status"] == "filtered_candidate"
    assert data["review_status"] == "unreviewed"
    assert data["human_audited"] is False
    assert data["safe_fallback_triggered"] is False
    assert data["component_statuses"]["generation"] == "SKIPPED"


def test_process_generation_fallback(mock_retriever):
    """Hate input below threshold routes to M9 fan-out -> M10 ranking -> M11 gate."""
    classifier = MockClassifier(default_is_hate=True, default_category="caste")
    pipeline = PipelineOrchestrator(
        classifier=classifier,
        router=FallbackRouter(retriever=mock_retriever, default_threshold=1.0001),
        fan_out=create_default_fan_out(use_mock=True),
        ranker=CandidateRanker(evaluator=MockCandidateEvaluator()),
        final_gate=FinalSafetyGate(),
        cleaner=TextCleaner(),
        allow_mock=True,
        threshold=1.0001
    )
    app = create_app(orchestrator=pipeline, test_config={"TESTING": True})
    client = app.test_client()

    res = client.post("/api/process", json={"text": "Completely novel and low similarity caste hate comment"})
    assert res.status_code == 200
    data = res.get_json()

    assert data["outcome"] == "GENERATED_RESPONSE"
    assert data["is_hate"] is True
    assert data["source_origin"] == "GENERATION"
    assert data["response_text"] is not None
    assert len(data["response_text"]) > 10
    assert data["quality_status"] == "generated_candidate"
    assert data["review_status"] == "unreviewed"
    assert data["human_audited"] is False
    assert data["safe_fallback_triggered"] is False
    assert data["component_statuses"]["router"] == "FALLBACK_GENERATION"
    assert data["component_statuses"]["generation"] == "SUCCESS"


def test_process_empty_input_safe_fallback(client):
    """Empty or whitespace text triggers fail-closed safe fallback with 200 OK."""
    res = client.post("/api/process", json={"text": "   "})
    assert res.status_code == 200
    data = res.get_json()

    assert data["outcome"] == "SAFE_FALLBACK"
    assert data["safe_fallback_triggered"] is True
    assert data["fallback_response"] == "I cannot respond to this respectfully."
    assert "EMPTY_INPUT" in data["fallback_reason"]


def test_all_five_dashboard_presets(client):
    """Verify that all five demonstration presets execute through their designated branch."""
    # Preset 1: Non-Hate
    r1 = client.post("/api/process", json={"text": "Have a wonderful, peaceful, and blessed day with your family!"})
    assert r1.status_code == 200
    d1 = r1.get_json()
    assert d1["outcome"] == "NON_HATE_EARLY_EXIT"
    assert d1["is_hate"] is False
    assert d1["safe_fallback_triggered"] is False
    assert d1["component_statuses"]["router"] == "SKIPPED"

    # Preset 2: Retrieval Match
    r2 = client.post("/api/process", json={"text": "Sample religious hostility query"})
    assert r2.status_code == 200
    d2 = r2.get_json()
    assert d2["outcome"] == "RETRIEVED_RESPONSE"
    assert d2["is_hate"] is True
    assert d2["source_origin"] == "RETRIEVAL"
    assert d2["component_statuses"]["generation"] == "SKIPPED"

    # Preset 3: Generation Fan-Out
    r3 = client.post("/api/process", json={"text": "This community is destroying traditional cultural values and should not be permitted here."})
    assert r3.status_code == 200
    d3 = r3.get_json()
    assert d3["outcome"] == "GENERATED_RESPONSE"
    assert d3["is_hate"] is True
    assert d3["source_origin"] == "GENERATION"
    assert d3["component_statuses"]["router"] == "FALLBACK_GENERATION"
    assert d3["component_statuses"]["generation"] == "SUCCESS"

    # Preset 4: Safety Rejection
    r4 = client.post("/api/process", json={"text": "You are an idiot and I hope you die."})
    assert r4.status_code == 200
    d4 = r4.get_json()
    assert d4["outcome"] == "SAFE_FALLBACK"
    assert d4["safe_fallback_triggered"] is True
    assert d4["component_statuses"]["final_gate"] == "REJECTED"
    assert "RETRIEVED_SAFETY_REJECTED" in d4["fallback_reason"]

    # Preset 5: Empty Input
    r5 = client.post("/api/process", json={"text": "   "})
    assert r5.status_code == 200
    d5 = r5.get_json()
    assert d5["outcome"] == "SAFE_FALLBACK"
    assert d5["safe_fallback_triggered"] is True
    assert d5["component_statuses"]["cleaner"] == "EMPTY_INPUT"


# =====================================================================
# Tests: KB Statistics & Review Queue Endpoints
# =====================================================================

def test_kb_stats_endpoint(client):
    """GET /api/kb/stats must return source candidate counts, review status, and categories."""
    res = client.get("/api/kb/stats")
    assert res.status_code == 200
    data = res.get_json()

    assert data["total_filtered_candidates"] == 19436
    assert data["review_queue_count"] == 28
    assert data["approved_kb_count"] == 0
    assert data["human_approved"] is False
    assert data["approval_status"] == "pending_human_review"
    assert "Religion" in data["categories"]
    assert "Gender" in data["categories"]


def test_kb_review_queue_endpoint(client):
    """GET /api/kb/review-queue must return the 28 candidates with their AI assessments."""
    res = client.get("/api/kb/review-queue")
    assert res.status_code == 200
    data = res.get_json()

    assert data["total_candidates"] == 28
    assert data["human_approved"] is False
    assert data["review_method"] == "ai_assisted"

    candidates = data["candidates"]
    assert len(candidates) == 28

    # Verify first candidate has assessment attached
    first = candidates[0]
    assert "kb_id" in first
    assert "hate_text" in first
    assert "counter_speech" in first
    assert "ai_assessment" in first
    assessment = first["ai_assessment"]
    assert assessment["recommendation"] in ["acceptable_with_caveats", "revise", "reject"]
    assert assessment["human_approved"] is False
