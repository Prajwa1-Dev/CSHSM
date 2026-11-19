"""
Formal M4 Semantic Retrieval Validation Test Suite.

Verifies:
1. Genuine BGE-M3 dense embeddings (1024 dimensions, unit L2 norms, absence of mock periodicity).
2. Query embedding parity and FAISS IndexFlatIP mathematical fidelity.
3. High-precision semantic retrieval across all 6 research categories (Gender, Political, Religion, Geo-political, Violence, Others).
4. Category filtering and absence of false retrievals on negative/out-of-domain queries at tau = 0.80.
5. Strict isolation between the 24 demo KB entries and the 1,171 unapproved candidates.
"""

import os
import sys
import json
import math
import pytest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.embedder import (
    BaseEmbedder,
    BGE_M3_Embedder,
    MockEmbedder,
    EmbeddingPipeline,
    load_npy_f32
)
from src.retrieval.retriever import (
    CSHSMRetriever,
    FAISSVectorIndex,
    ExactInnerProductIndex,
    RetrievalResult
)


@pytest.fixture(scope="module")
def bge_embedder():
    """Initializes real local BGE-M3 model on CUDA (or CPU fallback) in offline mode."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device=device,
        batch_size=4,
        local_files_only=True
    )


@pytest.fixture(scope="module")
def committed_retriever(bge_embedder):
    """Loads authoritative CSHSMRetriever over the committed demonstration KB."""
    return CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=bge_embedder,
        force_backend="faiss"
    )


# =====================================================================
# Test 1: Vector Matrix Mathematical Integrity
# =====================================================================

def test_demo_kb_vector_matrix_integrity():
    """
    Verifies that the 24 indexed entries have genuine float32 unit vectors
    from a neural model, rather than periodic mock hashes.
    """
    vector_file = "data/counter_speech_kb/embeddings/bge_m3_dense.npy"
    assert os.path.exists(vector_file), f"Vector file missing at {vector_file}"

    vectors = np.load(vector_file)
    assert vectors.shape == (24, 1024), f"Expected shape (24, 1024), got {vectors.shape}"
    assert vectors.dtype == np.float32, f"Expected float32, got {vectors.dtype}"

    # Verify unit norms
    norms = np.linalg.norm(vectors, axis=1)
    for idx, norm in enumerate(norms):
        assert abs(norm - 1.0) <= 1e-5, f"Vector {idx} norm is {norm}, expected 1.0"

    # Verify absence of MockEmbedder periodicity (mock hashes repeat with period 32)
    for row in vectors:
        assert not np.allclose(row[:32], row[32:64], atol=1e-3), (
            "Detected period-32 repeating pattern characteristic of MockEmbedder hash digest."
        )

    # Verify dense representation richness (real continuous embeddings have > 900 unique rounded floats)
    unique_counts = [len(np.unique(np.round(row, 5))) for row in vectors]
    assert min(unique_counts) > 900, (
        f"Minimum unique values per row is {min(unique_counts)}, expected > 900 for real neural embeddings."
    )


# =====================================================================
# Test 2: FAISS IndexFlatIP Mathematical Equivalence
# =====================================================================

def test_faiss_exact_ip_equivalence():
    """
    Verifies that FAISS IndexFlatIP produces identical top-k rankings
    and floating-point scores to exact linear algebra inner product.
    """
    vectors = np.load("data/counter_speech_kb/embeddings/bge_m3_dense.npy").tolist()

    faiss_index = FAISSVectorIndex(dim=1024)
    faiss_index.add(vectors)

    exact_index = ExactInnerProductIndex(dim=1024)
    exact_index.add(vectors)

    # Search with first vector as query
    q = [vectors[0]]
    f_scores, f_indices = faiss_index.search(q, k=5)
    e_scores, e_indices = exact_index.search(q, k=5)

    assert f_indices == e_indices, f"Index ranking mismatch: FAISS {f_indices} vs Exact {e_indices}"
    for fs, es in zip(f_scores[0], e_scores[0]):
        assert abs(fs - es) <= 1e-5, f"Score discrepancy {abs(fs - es)} exceeds tolerance"


# =====================================================================
# Test 3: Benchmark Retrieval Across All Six Categories
# =====================================================================

BENCHMARK_PARAPHRASES = [
    # Gender
    ("Gender", "DEMO-KB-00001", "Women shouldn't be allowed in leadership positions or tech companies, they only belong in the kitchen."),
    ("Gender", "DEMO-KB-00002", "Trans people have a severe mental disorder and need to be banned from all public spaces."),
    ("Gender", "DEMO-KB-00003", "Why waste money on higher education for girls when their only purpose is to marry and cook?"),
    # Political
    ("Political", "DEMO-KB-00005", "Anyone who supports the political opposition is a treasonous traitor who should be stripped of citizenship."),
    ("Political", "DEMO-KB-00006", "College activists protesting against the government need to be thrown in jail and beaten up."),
    ("Political", "DEMO-KB-00007", "Liberal professors and intellectuals are dangerous parasites destroying the moral fabric of our society."),
    # Religion
    ("Religion", "DEMO-KB-00009", "Every single Muslim is an undercover terrorist waiting to execute a holy war against our nation."),
    ("Religion", "DEMO-KB-00010", "Hindu idol worshippers with their primitive superstitious customs have ruined our civilized society."),
    ("Religion", "DEMO-KB-00012", "International banking cartels and mainstream media are secretly controlled by the Jews."),
    # Geo-political
    ("Geo-political", "DEMO-KB-00013", "Illegal immigrants and refugees are diseased parasites who commit violent crimes and steal our tax money."),
    ("Geo-political", "DEMO-KB-00014", "Migrants moving in from nearby provinces are just freeloading thieves ruining our hometown."),
    ("Geo-political", "DEMO-KB-00015", "All foreign migrant workers are taking jobs from citizens and should be immediately deported."),
    # Violence
    ("Violence", "DEMO-KB-00017", "Those street protestors should all be crushed with heavy bulldozers so they shut up forever."),
    ("Violence", "DEMO-KB-00018", "The court system is useless, we should grab guns and publicly lynch every one of these criminals."),
    ("Violence", "DEMO-KB-00019", "There will never be peace until we completely wipe out and exterminate every last person in that tribe."),
    # Others
    ("Others", "DEMO-KB-00021", "Lower-caste people are genetically inferior and should be banned from college admissions."),
    ("Others", "DEMO-KB-00022", "People born with physical or mental handicaps are worthless parasites consuming taxpayer resources."),
    ("Others", "DEMO-KB-00023", "Dark-skinned people are biologically uncivilized and only fit for manual street cleaning labor.")
]


def test_semantic_retrieval_across_all_six_categories(committed_retriever):
    """
    Verifies that paraphrased queries across all 6 categories achieve high
    top-1 accuracy (>= 90%) and strong semantic similarity scores.
    """
    correct_matches = 0
    scores = []

    for cat, expected_kb_id, query_text in BENCHMARK_PARAPHRASES:
        # Test category-filtered retrieval as orchestrated by the CSHSM pipeline
        result = committed_retriever.retrieve_by_text(query_text, top_k=3, category_filter=cat)
        assert len(result.matches) > 0, f"Retriever returned 0 matches for query: {query_text}"

        top1 = result.matches[0]
        scores.append(top1.score)

        if top1.kb_id == expected_kb_id:
            correct_matches += 1

        # Check category integrity
        assert top1.category == cat, f"Retrieved category {top1.category} != filtered category {cat}"
        # All genuine paraphrases should have strong semantic similarity
        assert top1.score >= 0.70, f"Score {top1.score} for query '{query_text}' is lower than expected"

    accuracy = correct_matches / len(BENCHMARK_PARAPHRASES)
    assert accuracy >= 0.90, f"Benchmark top-1 accuracy {accuracy:.2%} is below required 90%"


# =====================================================================
# Test 4: Negative & Out-of-Domain Separation from Threshold tau = 0.80
# =====================================================================

NEGATIVE_QUERIES = [
    "Cyclists on our city roads are arrogant jerks who intentionally block traffic and cause accidents.",
    "Elderly drivers over 70 are dangerous hazards who shouldn't have driver licenses.",
    "Modern video game players are anti-social degenerates ruining their lives in their basements.",
    "The municipal water supply has been contaminated due to pipe maintenance delays in sector 4.",
    "Artificial intelligence language models are undergoing unprecedented algorithmic improvements.",
    "What are the best tourist destinations to visit during autumn in Northern Europe?"
]


def test_negative_and_out_of_domain_queries_fall_below_tau(committed_retriever):
    """
    Verifies that unrelated / non-KB hate and benign queries fail to meet the
    tau = 0.80 threshold, ensuring zero false retrievals on out-of-domain inputs.
    """
    tau = 0.80
    max_negative_score = 0.0

    for neg_query in NEGATIVE_QUERIES:
        # Unconstrained retrieval: searches all 24 entries
        result = committed_retriever.retrieve_by_text(neg_query, top_k=1, category_filter=None)
        assert len(result.matches) > 0

        score = result.matches[0].score
        if score > max_negative_score:
            max_negative_score = score

        # Critical threshold assertion: negative queries must fall below tau
        assert score < tau, (
            f"Negative query '{neg_query}' scored {score:.4f}, which exceeds tau={tau} (False Retrieval)"
        )

    # Maximum observed negative query score should remain below 0.70
    assert max_negative_score < 0.70, f"Max negative score {max_negative_score} is too close to threshold"


# =====================================================================
# Test 5: Candidate Queue & Simulated Pool Strict Isolation
# =====================================================================

def test_candidate_queue_isolation():
    """
    Verifies that the 1,171 unapproved simulated candidates remain quarantined
    and completely isolated from the demonstration KB.
    """
    simulated_path = "data/counter_speech_kb/approved_kb_simulated.jsonl"
    demo_path = "data/counter_speech_kb/demo_approved_kb.jsonl"
    latest_pointer = "data/counter_speech_kb/embeddings/LATEST.json"

    assert os.path.exists(simulated_path)
    assert os.path.exists(demo_path)
    assert os.path.exists(latest_pointer)

    sim_count = sum(1 for line in open(simulated_path, "r", encoding="utf-8") if line.strip())
    demo_count = sum(1 for line in open(demo_path, "r", encoding="utf-8") if line.strip())

    assert sim_count == 1171, f"Expected 1,171 simulated candidates, found {sim_count}"
    assert demo_count == 24, f"Expected 24 demo candidates, found {demo_count}"

    # Verify pointer strictly points to 24-vector run
    with open(latest_pointer, "r", encoding="utf-8") as f:
        latest = json.load(f)
    assert latest["vector_count"] == 24
    assert latest["is_mock"] is False
    assert latest["model_identifier"] == "BAAI/bge-m3"
