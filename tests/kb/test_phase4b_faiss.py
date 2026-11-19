"""
Targeted tests for Phase 4B: FAISS Integration (IndexFlatIP) and Retrieval Architecture.

Covers:
1. FAISS index construction, dimension (1024), and ntotal.
2. L2 normalized vector dot-product search.
3. Mathematical and ranking consistency against ExactInnerProductIndex.
4. Category filtering in CSHSMRetriever with FAISS backend.
5. Empty index, non-positive k, dimension mismatch, and NaN/Inf guards.
6. CSHSMRetriever integration with real BGE-M3 embeddings on committed run.
7. Graceful fallback to ExactInnerProductIndex.
8. MockEmbedder compatibility.
9. Serialization (save and load) of FAISS index.
"""

import os
import sys
import math
import pytest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.embedder import (
    MockEmbedder,
    BGE_M3_Embedder,
    load_npy_f32
)
from src.retrieval.retriever import (
    BaseVectorIndex,
    FAISSVectorIndex,
    ExactInnerProductIndex,
    create_vector_index,
    CSHSMRetriever,
    RetrievalResult
)


@pytest.fixture(scope="module")
def bge_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device=device,
        batch_size=4,
        local_files_only=True
    )


def test_faiss_index_construction_and_dimension():
    """Verify FAISSVectorIndex initialization, dimension, and backend name."""
    idx = FAISSVectorIndex(dim=1024)
    assert idx.dimension == 1024
    assert idx.ntotal == 0
    assert idx.backend_name == "faiss"


def test_faiss_add_and_search_numerical_consistency():
    """Verify IndexFlatIP produces exact dot product on unit vectors."""
    idx = FAISSVectorIndex(dim=4)
    # 3 orthogonal unit vectors
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    idx.add(vectors)
    assert idx.ntotal == 3

    # Query identical to vector 0
    query = [[1.0, 0.0, 0.0, 0.0]]
    scores, indices = idx.search(query, k=3)

    assert indices[0][0] == 0
    assert abs(scores[0][0] - 1.0) < 1e-5
    assert abs(scores[0][1] - 0.0) < 1e-5
    assert abs(scores[0][2] - 0.0) < 1e-5


def test_faiss_ranking_and_scores_equal_to_exact_index():
    """Verify that FAISS and ExactInnerProductIndex produce identical ranking and scores."""
    dim = 64
    np.random.seed(42)
    # Generate 50 random unit vectors
    raw = np.random.randn(50, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = (raw / norms).tolist()

    faiss_idx = FAISSVectorIndex(dim=dim)
    exact_idx = ExactInnerProductIndex(dim=dim)
    faiss_idx.add(vectors)
    exact_idx.add(vectors)

    # 5 random query vectors
    q_raw = np.random.randn(5, dim).astype(np.float32)
    q_norms = np.linalg.norm(q_raw, axis=1, keepdims=True)
    queries = (q_raw / q_norms).tolist()

    f_scores, f_indices = faiss_idx.search(queries, k=5)
    e_scores, e_indices = exact_idx.search(queries, k=5)

    assert f_indices == e_indices
    for f_row, e_row in zip(f_scores, e_scores):
        for fs, es in zip(f_row, e_row):
            assert abs(fs - es) < 1e-5


def test_faiss_empty_input_and_error_guards():
    """Verify fail-closed validations on empty index, bad k, bad dimensions, and non-finite values."""
    idx = FAISSVectorIndex(dim=1024)

    # Empty index search guard
    with pytest.raises(ValueError, match="Cannot search an empty index"):
        idx.search([[0.1] * 1024], k=1)

    # Add valid vector
    idx.add([[0.1] * 1024])

    # Non-positive k
    with pytest.raises(ValueError, match="top_k must be positive"):
        idx.search([[0.1] * 1024], k=0)

    # Dimension mismatch in query
    with pytest.raises(ValueError, match="Query shape"):
        idx.search([[0.1] * 512], k=1)

    # Dimension mismatch in add
    with pytest.raises(ValueError, match="Vector shape"):
        idx.add([[0.1] * 512])

    # Non-finite in query
    bad_q = [0.1] * 1024
    bad_q[5] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        idx.search([bad_q], k=1)

    # Non-finite in add
    bad_v = [0.1] * 1024
    bad_v[10] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        idx.add([bad_v])


def test_faiss_retriever_committed_run_bge_m3(bge_embedder):
    """Verify CSHSMRetriever initializes with FAISS backend and retrieves correctly."""
    retriever = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=bge_embedder,
        force_backend="faiss"
    )

    assert retriever.index.backend_name == "faiss"
    assert retriever.index.ntotal == 24

    res = retriever.retrieve_by_text(
        "Females are incapable of technical leadership or engineering roles.",
        top_k=3
    )

    assert isinstance(res, RetrievalResult)
    assert res.backend == "faiss"
    assert res.is_mock is False
    assert len(res.matches) == 3
    assert res.matches[0].kb_id == "DEMO-KB-00001"
    assert res.matches[0].category == "Gender"
    assert res.matches[0].score >= 0.80


def test_faiss_category_filtering(bge_embedder):
    """Verify category filtering works seamlessly through FAISS backend."""
    retriever = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=bge_embedder,
        force_backend="faiss"
    )

    res = retriever.retrieve_by_text(
        "Citizens who cast their vote for opposition politicians are national traitors.",
        top_k=3,
        category_filter="Political"
    )

    assert len(res.matches) > 0
    for m in res.matches:
        assert m.category == "Political"
    assert res.matches[0].kb_id == "DEMO-KB-00005"
    assert res.matches[0].score >= 0.80


def test_faiss_fallback_to_exact_inner_product():
    """Verify factory create_vector_index respects force_backend and fallbacks."""
    # Explicit FAISS
    idx_faiss = create_vector_index(dim=1024, force_backend="faiss")
    assert idx_faiss.backend_name == "faiss"

    # Explicit Fallback
    idx_exact = create_vector_index(dim=1024, force_backend="exact_ip_fallback")
    assert idx_exact.backend_name == "exact_ip_fallback"

    # Default with prefer_faiss=True
    idx_default = create_vector_index(dim=1024, prefer_faiss=True)
    assert idx_default.backend_name == "faiss"

    # Default with prefer_faiss=False
    idx_no_faiss = create_vector_index(dim=1024, prefer_faiss=False)
    assert idx_no_faiss.backend_name == "exact_ip_fallback"


def test_faiss_with_mock_embedder():
    """Verify FAISS handles MockEmbedder vectors without errors."""
    mock_emb = MockEmbedder(dim=1024)
    vecs = mock_emb.embed_batch(["Comment A", "Comment B", "Comment C"])

    idx = FAISSVectorIndex(dim=1024)
    idx.add(vecs)
    assert idx.ntotal == 3

    q_vec = mock_emb.embed_batch(["Comment A"])[0]
    scores, indices = idx.search([q_vec], k=2)

    assert indices[0][0] == 0
    assert abs(scores[0][0] - 1.0) < 1e-4


def test_faiss_save_and_load(tmp_path):
    """Verify FAISS index serialization and deserialization."""
    fpath = str(tmp_path / "test_faiss.index")
    idx = FAISSVectorIndex(dim=4)
    idx.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    idx.save(fpath)

    assert os.path.exists(fpath)
    loaded_idx = FAISSVectorIndex.load(fpath, dim=4)
    assert loaded_idx.ntotal == 2
    assert loaded_idx.dimension == 4

    scores, indices = loaded_idx.search([[1.0, 0.0, 0.0, 0.0]], k=1)
    assert indices[0][0] == 0
    assert abs(scores[0][0] - 1.0) < 1e-5
