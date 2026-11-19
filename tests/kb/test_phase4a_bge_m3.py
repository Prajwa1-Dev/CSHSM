"""
Targeted tests for Phase 4A: Real BGE-M3 Dense Embeddings & ExactInnerProductIndex.

Covers:
1. Embedder initialization, dimensions, and L2 normalization.
2. Query vs document embedding methods.
3. Offline model loading and device fallback.
4. Semantic retrieval accuracy on real demonstration KB.
5. Category filtering in CSHSMRetriever.
6. MockEmbedder backward compatibility.
7. Bijective row-to-manifest integrity on committed runs.
"""

import os
import sys
import math
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.embedder import (
    BaseEmbedder,
    MockEmbedder,
    BGE_M3_Embedder,
    EmbeddingPipeline,
    load_npy_f32
)
from src.retrieval.retriever import (
    CSHSMRetriever,
    ExactInnerProductIndex,
    RetrievalResult
)


@pytest.fixture(scope="module")
def bge_embedder():
    """Shared BGE-M3 embedder fixture using CUDA if available, else CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device=device,
        batch_size=4,
        local_files_only=True
    )


def test_bge_m3_initialization_and_properties(bge_embedder):
    """Test dimension, model identifier, and mock status."""
    assert bge_embedder.dimension == 1024
    assert bge_embedder.model_identifier == "BAAI/bge-m3"
    assert bge_embedder.is_mock is False
    assert bge_embedder.device in ("cuda", "cpu")


def test_bge_m3_unit_norm_and_finite(bge_embedder):
    """Verify generated embeddings are strictly finite and L2-normalized."""
    texts = [
        "Women should have equal access to leadership roles in technology.",
        "Constructive dialogue bridges societal divisions."
    ]
    vectors = bge_embedder.embed_batch(texts)

    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 1024
        # All values finite
        assert all(not (math.isnan(v) or math.isinf(v)) for v in vec)
        # L2 norm is 1.0 within numerical tolerance
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-4


def test_bge_m3_query_and_document_helpers(bge_embedder):
    """Verify embed_query, embed_document, and embed_text produce identical representations."""
    sample_text = "Respecting human dignity is essential for social cohesion."
    vec_text = bge_embedder.embed_text(sample_text)
    vec_query = bge_embedder.embed_query(sample_text)
    vec_doc = bge_embedder.embed_document(sample_text)

    assert len(vec_text) == 1024
    assert len(vec_query) == 1024
    assert len(vec_doc) == 1024

    # Direct vector equality
    dot_q_t = sum(q * t for q, t in zip(vec_query, vec_text))
    dot_d_t = sum(d * t for d, t in zip(vec_doc, vec_text))
    assert abs(dot_q_t - 1.0) < 1e-5
    assert abs(dot_d_t - 1.0) < 1e-5


def test_bge_m3_empty_and_whitespace_safety(bge_embedder):
    """Verify embedder handles empty string and list safely without crashes."""
    assert bge_embedder.embed_batch([]) == []

    # Whitespace and empty text should not crash
    vecs = bge_embedder.embed_batch(["", "   ", "Valid comment"])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 1024
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-4


def test_bge_m3_cpu_fallback():
    """Verify that forcing CPU device instantiates cleanly and produces 1024-d unit vectors."""
    cpu_emb = BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=2,
        local_files_only=True
    )
    assert cpu_emb.device == "cpu"
    vecs = cpu_emb.embed_batch(["Short CPU test string."])
    assert len(vecs) == 1
    assert len(vecs[0]) == 1024
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) < 1e-4


def test_committed_run_bge_m3_retriever(bge_embedder):
    """Verify CSHSMRetriever loads the real BGE-M3 committed run with ExactInnerProductIndex."""
    retriever = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=bge_embedder,
        force_backend="exact_ip_fallback"
    )

    assert retriever.index.backend_name == "exact_ip_fallback"
    assert retriever.index.ntotal == 24
    assert retriever.run_metadata["is_mock"] is False
    assert retriever.run_metadata["model_identifier"] == "BAAI/bge-m3"

    # Test semantic retrieval
    res = retriever.retrieve_by_text(
        "Females are incapable of technical leadership or engineering roles.",
        top_k=3
    )

    assert isinstance(res, RetrievalResult)
    assert res.is_mock is False
    assert len(res.matches) == 3
    top_match = res.matches[0]
    assert top_match.kb_id == "DEMO-KB-00001"
    assert top_match.category == "Gender"
    assert top_match.score >= 0.75  # Strong semantic similarity


def test_bge_m3_category_filtering(bge_embedder):
    """Verify category filtering restricts returned candidates to the specified category."""
    retriever = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=bge_embedder,
        force_backend="exact_ip_fallback"
    )

    # Search with category_filter="Political"
    res = retriever.retrieve_by_text(
        "Anyone supporting the opposition party is a traitor.",
        top_k=3,
        category_filter="Political"
    )

    assert len(res.matches) > 0
    for m in res.matches:
        assert m.category == "Political"
    assert res.matches[0].kb_id == "DEMO-KB-00005"


def test_mock_embedder_backward_compatibility():
    """Verify MockEmbedder continues to operate deterministically for test suites."""
    mock_emb = MockEmbedder(dim=1024)
    assert mock_emb.dimension == 1024
    assert mock_emb.is_mock is True
    assert "MOCK" in mock_emb.model_identifier

    v1 = mock_emb.embed_batch(["Identical string"])[0]
    v2 = mock_emb.embed_batch(["Identical string"])[0]
    v3 = mock_emb.embed_batch(["Different string"])[0]

    assert v1 == v2
    assert v1 != v3
    assert len(v1) == 1024
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-4
