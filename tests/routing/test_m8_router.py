import os
import sys
import math
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.kb.schema import KBEntry
from src.retrieval.embedder import MockEmbedder, EmbeddingPipeline
from src.retrieval.retriever import (
    ExactInnerProductIndex,
    RetrievalMatch,
    RetrievalResult,
    CSHSMRetriever
)
from src.routing.router import (
    RoutingDecision,
    FallbackRouter,
    validate_threshold,
    calculate_calibration_metrics,
    optimize_threshold
)


# =====================================================================
# Test Fixtures & Helpers
# =====================================================================

def create_sample_kb_entry(
    kb_id="KB-00001",
    hate_text="Sample hate text",
    counter_speech="Sample counter speech",
    category="Religion",
    quality_status="filtered_candidate",
    review_status="unreviewed"
) -> KBEntry:
    return KBEntry(
        kb_id=kb_id,
        hate_text=hate_text,
        counter_speech=counter_speech,
        category=category,
        target="Islamophobia",
        strategy=None,
        language="en",
        source="conan",
        source_id="TEST-001",
        provenance="existing-dataset",
        generation_method="human-authored",
        quality_status=quality_status,
        review_status=review_status,
        review_score=None,
        human_audited=False,
        created_at="2026-10-22T00:00:00Z",
        version="1.0",
        notes=""
    )


@pytest.fixture
def mock_retriever_setup(tmp_path):
    """Builds a deterministic M7 retriever using committed M6 pipeline and MockEmbedder."""
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"

    entries = [
        create_sample_kb_entry("KB-00001", "Hate speech about religion", "Peaceful counter narrative 1", category="Religion"),
        create_sample_kb_entry("KB-00002", "Hate speech about gender", "Constructive counter narrative 2", category="Gender"),
        create_sample_kb_entry("KB-00003", "Hate speech about origin", "Supportive counter narrative 3", category="Nationality"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    embedder = MockEmbedder(dim=1024)
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    pipeline.run(str(input_file))

    retriever = CSHSMRetriever.from_committed_run(
        output_dir=out_dir,
        scope="development_provisional",
        allow_mock=True,
        embedder=embedder
    )
    return retriever, embedder


# =====================================================================
# 1. Routing Path Tests
# =====================================================================

def test_router_above_threshold_routes_to_retrieve(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    # Query with identical vector to KB-00001 (cosine similarity = 1.0)
    query_vec = embedder.embed_batch(["Hate speech about religion"])[0]
    decision = router.route_vector(query_vec, threshold=0.80)

    assert decision.action == "RETRIEVE"
    assert decision.fallback_reason is None
    assert decision.fallback_payload is None
    assert decision.threshold == 0.80
    assert decision.similarity_score is not None
    assert abs(decision.similarity_score - 1.0) < 1e-4
    assert decision.retrieved_candidate is not None
    assert decision.retrieved_candidate.kb_id == "KB-00001"
    assert "PROVISIONAL CANDIDATES" in decision.disclaimer


def test_router_exact_boundary_routes_to_retrieve(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    query_vec = embedder.embed_batch(["Hate speech about religion"])[0]
    # Fetch exact score first
    raw_res = retriever.retrieve_by_vector(query_vec, top_k=1)
    exact_score = raw_res.matches[0].score

    # Set threshold exactly to top score
    decision = router.route_vector(query_vec, threshold=exact_score)

    assert decision.action == "RETRIEVE"
    assert decision.fallback_reason is None
    assert decision.retrieved_candidate is not None
    assert decision.similarity_score == exact_score


def test_router_below_threshold_routes_to_fallback(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    # Orthogonal / different query
    query_vec = embedder.embed_batch(["Completely unrelated query about astronomy stars galaxies"])[0]
    # Set high threshold to force fallback
    decision = router.route_vector(query_vec, threshold=0.999)

    assert decision.action == "FALLBACK_GENERATION"
    assert decision.fallback_reason == "LOW_SIMILARITY"
    assert decision.retrieved_candidate is None
    assert decision.similarity_score is not None
    assert decision.fallback_payload is not None
    assert "top_candidate" in decision.fallback_payload
    assert decision.fallback_payload["threshold"] == 0.999
    assert "SAFE FALLBACK" in decision.disclaimer


def test_router_empty_retrieval_routes_to_fallback(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    query_vec = embedder.embed_batch(["Hate speech about religion"])[0]
    # Filter with non-existent category forces matches = []
    decision = router.route_vector(query_vec, category_filter="NonExistentCategory")

    assert decision.action == "FALLBACK_GENERATION"
    assert decision.fallback_reason == "EMPTY_RETRIEVAL"
    assert decision.retrieved_candidate is None
    assert decision.similarity_score is None
    assert decision.fallback_payload["matches_count"] == 0


def test_router_retriever_exception_routes_to_fallback(mock_retriever_setup):
    retriever, _ = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    # Pass invalid dimension vector (e.g. 512 instead of 1024), which triggers ValueError in index
    bad_vector = [0.1] * 512
    decision = router.route_vector(bad_vector)

    assert decision.action == "FALLBACK_GENERATION"
    assert decision.fallback_reason == "RETRIEVER_ERROR"
    assert decision.retrieved_candidate is None
    assert decision.similarity_score is None
    assert decision.fallback_payload["error_type"] == "ValueError"
    assert "SAFE FALLBACK" in decision.disclaimer


def test_router_invalid_threshold_routes_to_fallback(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    query_vec = embedder.embed_batch(["Hate speech about religion"])[0]

    # NaN threshold
    decision_nan = router.route_vector(query_vec, threshold=float("nan"))
    assert decision_nan.action == "FALLBACK_GENERATION"
    assert decision_nan.fallback_reason == "INVALID_THRESHOLD"

    # Out of range threshold (> 1.05)
    decision_oor = router.route_vector(query_vec, threshold=2.0)
    assert decision_oor.action == "FALLBACK_GENERATION"
    assert decision_oor.fallback_reason == "INVALID_THRESHOLD"

    # Non-numeric string threshold
    decision_str = router.route_vector(query_vec, threshold="invalid_threshold")  # type: ignore
    assert decision_str.action == "FALLBACK_GENERATION"
    assert decision_str.fallback_reason == "INVALID_THRESHOLD"


def test_router_route_text_end_to_end(mock_retriever_setup):
    retriever, _ = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    # Route text with matching phrase
    decision = router.route_text("Hate speech about religion", threshold=0.80)
    assert decision.action == "RETRIEVE"
    assert decision.retrieved_candidate.kb_id == "KB-00001"
    assert decision.query == "Hate speech about religion"

    # Route text with high threshold (tau* = 1.0001 forces fallback even for identical match)
    decision_fb = router.route_text("Hate speech about religion", threshold=1.0001)
    assert decision_fb.action == "FALLBACK_GENERATION"
    assert decision_fb.fallback_reason == "LOW_SIMILARITY"

    # Route text with unrelated phrase and threshold 0.80
    decision_unrel = router.route_text("Completely unrelated text about stars and astronomy", threshold=0.80)
    assert decision_unrel.action == "FALLBACK_GENERATION"
    assert decision_unrel.fallback_reason == "LOW_SIMILARITY"


def test_router_route_text_without_embedder_fails_closed():
    # Construct retriever with embedder=None
    index = ExactInnerProductIndex(dim=4)
    index.add([[1.0, 0.0, 0.0, 0.0]])
    manifest = [{"row_idx": 0, "kb_id": "KB-1", "hate_text": "text", "counter_speech": "cs", "quality_status": "filtered_candidate", "review_status": "unreviewed"}]
    retriever = CSHSMRetriever(index=index, manifest=manifest, run_metadata={"run_id": "TEST", "scope": "development_provisional", "is_mock": True}, embedder=None)

    router = FallbackRouter(retriever=retriever, default_threshold=0.80)
    decision = router.route_text("Sample text query")

    assert decision.action == "FALLBACK_GENERATION"
    assert decision.fallback_reason == "RETRIEVER_ERROR"
    assert "CSHSMRetriever has no embedder configured" in decision.fallback_payload["error_message"]


def test_router_initialization_validations(mock_retriever_setup):
    retriever, _ = mock_retriever_setup

    with pytest.raises(ValueError, match="retriever cannot be None"):
        FallbackRouter(retriever=None)

    with pytest.raises(ValueError, match="Threshold cannot be NaN or Inf"):
        FallbackRouter(retriever=retriever, default_threshold=float("nan"))

    with pytest.raises(ValueError, match="Threshold must be in"):
        FallbackRouter(retriever=retriever, default_threshold=1.5)


# =====================================================================
# 2. Provenance & Disclaimer Preservation Tests
# =====================================================================

def test_unapproved_candidate_disclaimer_and_provenance_preservation(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup
    router = FallbackRouter(retriever=retriever, default_threshold=0.80)

    query_vec = embedder.embed_batch(["Hate speech about religion"])[0]
    decision = router.route_vector(query_vec, threshold=0.80)

    # Verify unapproved candidate safeguards
    cand = decision.retrieved_candidate
    assert cand is not None
    assert cand.quality_status == "filtered_candidate"
    assert cand.review_status == "unreviewed"
    assert cand.scope == "development_provisional"

    # Verify disclaimer is preserved verbatim
    assert decision.disclaimer == (
        "PROVISIONAL CANDIDATES — Counter-speech retrieved for retrieval engineering only; "
        "unreviewed and unapproved by M5.4."
    )

    # Verify provenance
    assert decision.provenance["threshold_applied"] == 0.80
    assert decision.provenance["routing_action"] == "RETRIEVE"
    assert decision.provenance["run_id"] == retriever.run_metadata["run_id"]

    # Verify serialization
    d = decision.to_dict()
    assert d["action"] == "RETRIEVE"
    assert d["threshold"] == 0.80
    assert d["retrieved_candidate"]["kb_id"] == "KB-00001"
    json_str = json.dumps(d)
    assert "KB-00001" in json_str


# =====================================================================
# 3. Calibration Metrics Tests
# =====================================================================

def test_calibration_metrics_standard():
    scores = [0.90, 0.85, 0.80, 0.75, 0.60]
    labels = [1, 1, 0, 1, 0]
    safety_labels = [1, 1, 1, 0, 1]

    # Evaluate at tau = 0.80 (retrieves scores >= 0.80: [0.90, 0.85, 0.80])
    m = calculate_calibration_metrics(
        scores=scores,
        labels=labels,
        safety_labels=safety_labels,
        threshold=0.80
    )

    assert m["threshold"] == 0.80
    assert m["n_total"] == 5
    assert m["n_retrieved"] == 3
    assert m["n_fallback"] == 2
    assert m["n_relevant_retrieved"] == 2  # labels at indices 0, 1 are 1; index 2 is 0
    assert m["n_safe_retrieved"] == 3      # safety labels at 0, 1, 2 are all 1
    assert m["coverage"] == 0.60           # 3 / 5
    assert m["fallback_rate"] == 0.40      # 2 / 5
    assert abs(m["precision"] - (2 / 3)) < 1e-5
    assert m["safety_rate"] == 1.00        # 3 / 3
    assert m["yield"] == 0.40              # 2 / 5


def test_calibration_metrics_zero_retrieved_handles_none():
    scores = [0.85, 0.75, 0.65]
    labels = [1, 1, 0]
    safety_labels = [1, 1, 1]

    # Threshold higher than all scores
    m = calculate_calibration_metrics(
        scores=scores,
        labels=labels,
        safety_labels=safety_labels,
        threshold=0.90
    )

    assert m["n_retrieved"] == 0
    assert m["n_fallback"] == 3
    assert m["coverage"] == 0.0
    assert m["fallback_rate"] == 1.0
    assert m["yield"] == 0.0
    # Must be None, NEVER 1.0 or 0.0
    assert m["precision"] is None
    assert m["safety_rate"] is None


def test_calibration_metrics_all_retrieved():
    scores = [0.85, 0.75, 0.65]
    labels = [1, 0, 1]
    safety_labels = [1, 1, 0]

    # Threshold lower than all scores
    m = calculate_calibration_metrics(
        scores=scores,
        labels=labels,
        safety_labels=safety_labels,
        threshold=0.50
    )

    assert m["n_retrieved"] == 3
    assert m["n_fallback"] == 0
    assert m["coverage"] == 1.0
    assert m["fallback_rate"] == 0.0
    assert abs(m["precision"] - (2 / 3)) < 1e-5
    assert abs(m["safety_rate"] - (2 / 3)) < 1e-5
    assert abs(m["yield"] - (2 / 3)) < 1e-5


def test_calibration_metrics_without_safety_labels():
    scores = [0.90, 0.70]
    labels = [1, 0]

    m = calculate_calibration_metrics(scores=scores, labels=labels, threshold=0.80)
    assert m["n_retrieved"] == 1
    assert m["precision"] == 1.0
    assert m["safety_rate"] is None
    assert m["n_safe_retrieved"] is None


def test_calibration_metrics_validation_errors():
    with pytest.raises(ValueError, match="Dimension mismatch: scores"):
        calculate_calibration_metrics([0.8], [1, 0])

    with pytest.raises(ValueError, match="Dimension mismatch: safety_labels"):
        calculate_calibration_metrics([0.8], [1], safety_labels=[1, 0])

    with pytest.raises(ValueError, match="not a finite numeric"):
        calculate_calibration_metrics([float("nan")], [1])

    with pytest.raises(ValueError, match="must be binary"):
        calculate_calibration_metrics([0.8], [2])

    with pytest.raises(ValueError, match="Threshold must be in"):
        calculate_calibration_metrics([0.8], [1], threshold=5.0)


def test_calibration_metrics_empty_dataset():
    m = calculate_calibration_metrics([], [])
    assert m["n_total"] == 0
    assert m["n_retrieved"] == 0
    assert m["coverage"] == 0.0
    assert m["fallback_rate"] == 1.0
    assert m["precision"] is None
    assert m["safety_rate"] is None
    assert m["yield"] == 0.0


# =====================================================================
# 4. Threshold Optimizer Tests
# =====================================================================

def test_optimizer_reachable_targets():
    scores = [0.95, 0.90, 0.85, 0.80, 0.75, 0.65]
    labels = [1, 1, 1, 0, 1, 0]
    safety_labels = [1, 1, 1, 1, 1, 0]

    # With target_precision=0.85:
    # tau=0.85 retrieves scores [0.95, 0.90, 0.85] -> 3 items, all labels=1 -> precision=1.00 (feasible, coverage=0.50)
    # tau=0.80 retrieves 4 items, labels=[1, 1, 1, 0] -> precision=0.75 (< 0.85, infeasible)
    # tau=0.75 retrieves 5 items, labels=[1, 1, 1, 0, 1] -> precision=4/5 = 0.80 (< 0.85, infeasible)
    # Optimal threshold maximizing coverage subject to precision >= 0.85 is 0.85
    res = optimize_threshold(
        scores=scores,
        labels=labels,
        safety_labels=safety_labels,
        min_threshold=0.60,
        max_threshold=0.95,
        step=0.05,
        target_precision=0.85,
        required_safety=1.00
    )

    assert res["status"] == "OPTIMAL"
    assert res["optimal_threshold"] == 0.85
    assert res["achieved_metrics"]["precision"] == 1.0
    assert res["achieved_metrics"]["coverage"] == 0.50
    assert "CALIBRATION PROTOCOL ONLY" in res["disclaimer"]

    # With target_precision=0.80:
    # tau=0.75 retrieves 5 items with precision 4/5 = 0.80 (>= 0.80) and higher coverage (5/6 = 0.833)
    res_80 = optimize_threshold(
        scores=scores,
        labels=labels,
        safety_labels=safety_labels,
        min_threshold=0.60,
        max_threshold=0.95,
        step=0.05,
        target_precision=0.80,
        required_safety=1.00
    )
    assert res_80["status"] == "OPTIMAL"
    assert res_80["optimal_threshold"] == 0.75
    assert res_80["achieved_metrics"]["precision"] == 0.80
    assert abs(res_80["achieved_metrics"]["coverage"] - (5 / 6)) < 1e-5


def test_optimizer_unreachable_target_fails_closed():
    scores = [0.90, 0.85, 0.80]
    labels = [0, 0, 0]  # No relevant items

    res = optimize_threshold(
        scores=scores,
        labels=labels,
        target_precision=0.80
    )

    assert res["status"] == "FAILED_CONSTRAINTS_FAIL_CLOSED"
    assert res["optimal_threshold"] == 1.0001
    assert res["achieved_metrics"]["coverage"] == 0.0
    assert res["achieved_metrics"]["fallback_rate"] == 1.0
    assert res["achieved_metrics"]["precision"] is None


def test_optimizer_unreachable_safety_fails_closed():
    scores = [0.95, 0.90]
    labels = [1, 1]
    safety_labels = [0, 0]  # All items unsafe

    res = optimize_threshold(
        scores=scores,
        labels=labels,
        safety_labels=safety_labels,
        required_safety=1.00
    )

    assert res["status"] == "FAILED_CONSTRAINTS_FAIL_CLOSED"
    assert res["optimal_threshold"] == 1.0001


def test_optimizer_boundary_tie_handling():
    # Only one high scoring candidate (score 0.90, label 1)
    # All other candidates have score 0.50
    scores = [0.90, 0.50]
    labels = [1, 0]

    # Grid points in [0.70, 0.85] (e.g. 0.70, 0.75, 0.80, 0.85) all retrieve only the top candidate
    # They have identical coverage (0.50), precision (1.0), and yield (0.50).
    res = optimize_threshold(
        scores=scores,
        labels=labels,
        min_threshold=0.70,
        max_threshold=0.85,
        step=0.05,
        target_precision=0.80
    )

    assert res["status"] == "OPTIMAL"
    # Deterministic tie-breaker selects the highest (most conservative) threshold
    assert res["optimal_threshold"] == 0.85


def test_optimizer_empty_dataset_fails_closed():
    res = optimize_threshold(scores=[], labels=[])
    assert res["status"] == "FAILED_CONSTRAINTS_FAIL_CLOSED"
    assert res["optimal_threshold"] == 1.0001


def test_optimizer_objective_yield_vs_coverage():
    # Score 0.90 (rel=1), 0.80 (rel=1), 0.70 (rel=0)
    scores = [0.90, 0.80, 0.70]
    labels = [1, 1, 0]

    # Maximize coverage with precision >= 0.60
    # tau=0.70 gives coverage 1.0, precision 2/3 = 0.667, yield 2/3 = 0.667
    res_cov = optimize_threshold(
        scores=scores,
        labels=labels,
        min_threshold=0.70,
        max_threshold=0.90,
        step=0.10,
        target_precision=0.60,
        objective="coverage"
    )
    assert res_cov["optimal_threshold"] == 0.70
    assert res_cov["achieved_metrics"]["coverage"] == 1.0


def test_fail_closed_threshold_enforces_100_percent_fallback_in_router(mock_retriever_setup):
    retriever, embedder = mock_retriever_setup

    # Query with exact match to index item (cosine similarity = 1.000000)
    query_vec = embedder.embed_batch(["Hate speech about religion"])[0]

    # Router configured with fail-closed threshold tau* = 1.0001
    router = FallbackRouter(retriever=retriever, default_threshold=1.0001)
    decision = router.route_vector(query_vec)

    # 1.000000 < 1.0001 -> MUST fallback to generation
    assert decision.action == "FALLBACK_GENERATION"
    assert decision.fallback_reason == "LOW_SIMILARITY"
    assert decision.retrieved_candidate is None
    assert decision.similarity_score is not None
    assert decision.similarity_score <= 1.000001
    assert decision.threshold == 1.0001
