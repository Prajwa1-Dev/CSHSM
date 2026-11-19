"""
Integration test suite for Knowledge Base V2 in CSHSM.

Verifies:
1. 300 x 1024 vector matrix shape, unit L2 norms, and manifest bijection.
2. Category balance (50 records per category across all 6 categories).
3. Human-audit invariant preservation (human_audited=False, quality_status='pending_review').
4. Category-wise retrieval fidelity across all six categories using FAISS IndexFlatIP.
5. Calibrated threshold routing at tau = 0.80 (in-domain -> RETRIEVE, out-of-domain -> FALLBACK).
6. Safety-gate handling (unreviewed candidate flagged as unapproved).
7. Strict isolation: Demo KB at data/counter_speech_kb/embeddings remains 24 vectors.
"""

import os
import sys
import json
import math
import pytest
import numpy as np

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.embedder import BGE_M3_Embedder, load_npy_f32
from src.retrieval.retriever import CSHSMRetriever, RetrievalResult
from src.routing.router import FallbackRouter, RoutingDecision
from src.safety.final_gate import FinalSafetyGate, FinalGateResult


V2_EMBEDDINGS_DIR = "data/counter_speech_kb/embeddings_v2"
DEMO_EMBEDDINGS_DIR = "data/counter_speech_kb/embeddings"


@pytest.fixture(scope="module")
def v2_embedder():
    """Shared BGE-M3 embedder in offline mode."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device=device,
        local_files_only=True
    )


@pytest.fixture(scope="module")
def v2_retriever(v2_embedder):
    """CSHSMRetriever loaded from committed V2 run."""
    return CSHSMRetriever.from_committed_run(
        output_dir=V2_EMBEDDINGS_DIR,
        allow_mock=False,
        embedder=v2_embedder
    )


def test_v2_vector_matrix_and_manifest_integrity():
    """Verify 300 x 1024 matrix, unit L2 norms, and manifest bijection."""
    latest_file = os.path.join(V2_EMBEDDINGS_DIR, "LATEST.json")
    assert os.path.exists(latest_file), f"LATEST.json missing in {V2_EMBEDDINGS_DIR}"

    with open(latest_file, "r", encoding="utf-8") as f:
        latest = json.load(f)

    assert latest["vector_count"] == 300
    assert latest["manifest_count"] == 300
    assert latest["matrix_shape"] == [300, 1024]
    assert latest["scope"] == "development_provisional"
    assert latest["is_mock"] is False
    assert latest["model_identifier"] == "BAAI/bge-m3"

    run_dir = os.path.join(V2_EMBEDDINGS_DIR, latest["run_dir"])
    vec_path = os.path.join(run_dir, "bge_m3_dense.npy")
    manifest_path = os.path.join(run_dir, "bge_m3_manifest.jsonl")

    vectors, shape = load_npy_f32(vec_path)
    assert shape == (300, 1024), f"Expected (300, 1024), got {shape}"

    # L2 norm check
    for idx, v in enumerate(vectors):
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-4, f"Vector {idx} norm {norm:.6f} != 1.0"

    # Manifest check
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 300
    cat_counts = {}
    for r in records:
        cat = r["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        assert r["quality_status"] == "pending_review"
        assert r["scope"] == "development_provisional"

    expected_categories = {"Gender", "Political", "Religion", "Geo-political", "Violence", "Others"}
    assert set(cat_counts.keys()) == expected_categories
    for cat in expected_categories:
        assert cat_counts[cat] == 50, f"Expected 50 entries for {cat}, got {cat_counts[cat]}"


def test_v2_category_wise_semantic_retrieval(v2_retriever):
    """
    Test 1 query per category in Romanized Kannada against the 300-entry KB V2.
    Asserts top-1 category matches query category and records similarity.
    """
    test_queries = [
        ("Gender", "Hudugir ge driving baralla road alli bandre accident guarantee", "CSHSM-KN-00001"),
        ("Political", "Ee party ge vote hakidavru ellaru murkharu, avarige yochane illa", "CSHSM-KN-00051"),
        ("Religion", "Ella Muslims terrorists, avarannu namma area ge bidabaaradu", "CSHSM-KN-00101"),
        ("Geo-political", "Bere rajyadavaru namma Bengaluru ge bandu namma kelsa kadiyuttare", "CSHSM-KN-00151"),
        ("Violence", "Kalla sikkidre road alli hodidu bidbeku, police ge kododu waste", "CSHSM-KN-00201"),
        ("Others", "Avanu kappu, avanannu yaaru preethisalla, makeup haaki bidu", "CSHSM-KN-00251"),
    ]

    for expected_cat, query_text, expected_id in test_queries:
        res: RetrievalResult = v2_retriever.retrieve_by_text(query_text, top_k=3)
        assert len(res.matches) > 0, f"No matches returned for query: {query_text}"
        top1 = res.matches[0]
        assert top1.category == expected_cat, (
            f"Category mismatch for query '{query_text}': Expected {expected_cat}, got {top1.category}"
        )
        assert top1.kb_id == expected_id, (
            f"ID mismatch for query '{query_text}': Expected {expected_id}, got {top1.kb_id}"
        )
        assert top1.score >= 0.85, (
            f"Low similarity score {top1.score:.4f} for query: {query_text}"
        )


def test_v2_threshold_routing_tau_0_80(v2_retriever):
    """
    Verify routing decisions at calibrated threshold tau = 0.80:
    - High-similarity in-domain query -> RETRIEVE (score >= 0.80)
    - Out-of-domain novel query -> FALLBACK_GENERATION (score < 0.80)
    """
    router = FallbackRouter(retriever=v2_retriever, default_threshold=0.80)

    # In-domain query (should exceed 0.80)
    in_domain_query = "Hudugir ge driving baralla road alli bandre accident guarantee"
    decision: RoutingDecision = router.route_text(in_domain_query)
    assert decision.action == "RETRIEVE", (
        f"Expected route RETRIEVE for in-domain query, got {decision.action} (score: {decision.similarity_score})"
    )
    assert decision.similarity_score >= 0.80
    assert decision.retrieved_candidate is not None

    # Out-of-domain query (should fall below 0.80)
    out_domain_query = "Today is a beautiful sunny morning to walk in the park with friends"
    decision_ood: RoutingDecision = router.route_text(out_domain_query)
    assert decision_ood.action == "FALLBACK_GENERATION", (
        f"Expected route FALLBACK_GENERATION for out-of-domain query, got {decision_ood.action} (score: {decision_ood.similarity_score})"
    )
    assert decision_ood.similarity_score is None or decision_ood.similarity_score < 0.80


def test_v2_safety_gate_flags_unapproved(v2_retriever):
    """
    Verify that counter-speech retrieved from V2 (which has human_audited=False)
    passes safety validation as unapproved provisional content (decision='PASS_UNAPPROVED').
    """
    gate = FinalSafetyGate(fallback_message="Default safe response")
    res = v2_retriever.retrieve_by_text("Hudugir ge driving baralla", top_k=1)
    match = res.matches[0]

    # Verify candidate properties
    assert match.scope == "development_provisional"
    assert match.quality_status == "pending_review"

    gate_res: FinalGateResult = gate.validate_retrieved_candidate(
        retrieval_match=match,
        allow_mock=True
    )
    # The safety gate allows non-toxic counter-speech but flags it as PASS_UNAPPROVED
    assert gate_res.decision == "PASS_UNAPPROVED"
    assert gate_res.human_audited is False
    assert gate_res.review_status == "unreviewed"
    assert "PROVISIONAL" in gate_res.disclaimer or "unapproved" in gate_res.disclaimer.lower()


def test_demo_kb_isolation_preserved():
    """
    Verify strict isolation: Demo KB at data/counter_speech_kb/embeddings
    remains at 24 vectors and is unaffected by V2.
    """
    latest_file = os.path.join(DEMO_EMBEDDINGS_DIR, "LATEST.json")
    assert os.path.exists(latest_file), f"Demo LATEST.json missing in {DEMO_EMBEDDINGS_DIR}"

    with open(latest_file, "r", encoding="utf-8") as f:
        demo_latest = json.load(f)

    assert demo_latest["vector_count"] == 24, f"Demo KB corrupted! Expected 24 vectors, got {demo_latest['vector_count']}"
    assert demo_latest["matrix_shape"] == [24, 1024]


def test_v2_category_filter_expanded_candidate_pool(v2_retriever):
    """
    Verifies that when category filtering is applied, retriever queries a larger
    candidate pool (search_k = min(ntotal, max(50, top_k * 20))), allowing
    relevant category matches that rank outside the global top 5 to survive.
    """
    # Query whose nearest global neighbors in FAISS are in Violence/Others (outside top 5 for Geo-political)
    query = "Bere state inda bandavru namma city ge burden, avarige illi place kodbardu."

    # 1. Category filtered retrieval with Geo-political
    res = v2_retriever.retrieve_by_text(query_text=query, top_k=1, category_filter="Geo-political")
    assert res.retrieval_status == "success"
    assert len(res.matches) == 1
    match = res.matches[0]
    assert match.category.lower() == "geo-political"
    assert match.score > 0.0

    # 2. Verify router evaluates candidate score against tau = 0.80 and routes to below threshold fallback
    router = FallbackRouter(retriever=v2_retriever, default_threshold=0.80)
    decision = router.route_text(query_text=query, category_filter="Geo-political", threshold=0.80)
    assert decision.action == "FALLBACK_GENERATION"
    assert decision.fallback_reason == "LOW_SIMILARITY"
    assert decision.similarity_score == match.score
    assert decision.similarity_score < 0.80
