import os
import sys
import json
import math
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.kb.schema import KBEntry
from src.retrieval.embedder import (
    BaseEmbedder,
    MockEmbedder,
    EmbeddingPipeline,
    save_npy_f32
)
from src.retrieval.retriever import (
    BaseVectorIndex,
    FAISSVectorIndex,
    ExactInnerProductIndex,
    create_vector_index,
    RetrievalMatch,
    RetrievalResult,
    CSHSMRetriever
)

def create_sample_kb_entry(
    kb_id="KB-00001",
    hate_text="Sample hate speech text",
    counter_speech="Sample constructive counter narrative",
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

def test_exact_inner_product_index_mathematics():
    index = ExactInnerProductIndex(dim=4)
    assert index.dimension == 4
    assert index.backend_name == "exact_ip_fallback"

    # Add 3 unit vectors
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    v3 = [0.707106, 0.707106, 0.0, 0.0]
    index.add([v1, v2, v3])
    assert index.ntotal == 3

    # Query with vector identical to v1
    q = [1.0, 0.0, 0.0, 0.0]
    scores, indices = index.search([q], k=3)

    assert len(scores[0]) == 3
    assert len(indices[0]) == 3

    # Rank 1 must be v1 with score 1.0
    assert indices[0][0] == 0
    assert abs(scores[0][0] - 1.0) < 1e-4

    # Rank 2 must be v3 with score ~0.7071
    assert indices[0][1] == 2
    assert abs(scores[0][1] - 0.707106) < 1e-4

    # Rank 3 must be v2 with score 0.0
    assert indices[0][2] == 1
    assert abs(scores[0][2] - 0.0) < 1e-4

def test_retriever_initialization_from_committed_run(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"

    entries = [
        create_sample_kb_entry("KB-00001", "Hate speech 1", "Counter narrative 1", category="Religion"),
        create_sample_kb_entry("KB-00002", "Hate speech 2", "Counter narrative 2", category="Gender"),
        create_sample_kb_entry("KB-00003", "Hate speech 3", "Counter narrative 3", category="Religion"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    # Produce committed run via M6 pipeline
    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    pipeline.run(str(input_file))

    # Initialize M7 retriever
    retriever = CSHSMRetriever.from_committed_run(
        output_dir=out_dir,
        scope="development_provisional",
        allow_mock=True
    )
    assert retriever.index.ntotal == 3
    assert len(retriever.manifest) == 3
    assert retriever.run_metadata["is_mock"] is True

def test_retrieval_by_vector_exact_match(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"

    entries = [
        create_sample_kb_entry("KB-00001", "Comment alpha", "Counter alpha"),
        create_sample_kb_entry("KB-00002", "Comment beta", "Counter beta"),
        create_sample_kb_entry("KB-00003", "Comment gamma", "Counter gamma"),
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

    # Query with exact vector of Comment beta
    target_vector = embedder.embed_batch(["Comment beta"])[0]
    result = retriever.retrieve_by_vector(target_vector, top_k=2)

    assert result.retrieval_status == "success"
    assert len(result.matches) == 2

    # Top match must be KB-00002 with cosine similarity ~1.0
    top = result.matches[0]
    assert top.rank == 1
    assert top.kb_id == "KB-00002"
    assert top.hate_text == "Comment beta"
    assert top.counter_speech == "Counter beta"
    assert abs(top.score - 1.0) < 1e-4

def test_retrieval_by_text(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"

    entries = [
        create_sample_kb_entry("KB-00001", "Text alpha", "Response alpha"),
        create_sample_kb_entry("KB-00002", "Text beta", "Response beta"),
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
        allow_mock=True,
        embedder=embedder
    )

    result = retriever.retrieve_by_text("Text alpha", top_k=1)
    assert result.query_text == "Text alpha"
    assert len(result.matches) == 1
    assert result.matches[0].kb_id == "KB-00001"

def test_retrieval_category_filtering(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"

    entries = [
        create_sample_kb_entry("KB-00001", "Religious comment", "Counter 1", category="Religion"),
        create_sample_kb_entry("KB-00002", "Gender comment", "Counter 2", category="Gender"),
        create_sample_kb_entry("KB-00003", "Another religious comment", "Counter 3", category="Religion"),
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

    retriever = CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True, embedder=embedder)

    # Filter by Gender
    res_gender = retriever.retrieve_by_text("Any query", top_k=2, category_filter="Gender")
    assert all(m.category == "Gender" for m in res_gender.matches)
    assert len(res_gender.matches) == 1
    assert res_gender.matches[0].kb_id == "KB-00002"

    # Filter by Religion
    res_rel = retriever.retrieve_by_text("Any query", top_k=2, category_filter="Religion")
    assert all(m.category == "Religion" for m in res_rel.matches)
    assert len(res_rel.matches) == 2

def test_retriever_fails_closed_on_uncommitted_run(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    runs_dir = os.path.join(out_dir, "runs")
    uncommitted_dir = os.path.join(runs_dir, "run_uncommitted")
    os.makedirs(uncommitted_dir, exist_ok=True)

    # Point pointer to uncommitted run
    with open(os.path.join(out_dir, "LATEST.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": "uncommitted", "run_dir": "runs/run_uncommitted"}, f)

    with pytest.raises(RuntimeError, match="uncommitted"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)

def test_retriever_fails_closed_on_mock_in_approved_scope(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    pipeline.run(str(input_file))

    # Must fail closed if allow_mock is False
    with pytest.raises(ValueError, match="Refusing to consume mock embeddings"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=False)

def test_retriever_fails_closed_on_scope_mismatch(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    pipeline.run(str(input_file))

    with pytest.raises(ValueError, match="Scope mismatch"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, scope="approved_kb", allow_mock=True)

def test_retriever_fails_closed_on_manifest_vector_length_mismatch(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Text 1", "Counter 1"),
        create_sample_kb_entry("KB-00002", "Text 2", "Counter 2"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    report = pipeline.run(str(input_file))

    # Corrupt the run directory by appending an extra line to manifest
    run_dir = os.path.join(out_dir, report["run_dir"])
    manifest_path = os.path.join(run_dir, report["artifacts"]["manifest_file"])
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"row_idx": 2, "kb_id": "KB-EXTRA"}) + "\n")

    # Length mismatch or hash corruption must raise
    with pytest.raises((RuntimeError, ValueError)):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)

def test_retriever_fails_closed_on_corrupted_artifact_hash(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    report = pipeline.run(str(input_file))

    # Tamper with byte in vector file in the committed run_dir
    run_dir = os.path.join(out_dir, report["run_dir"])
    vec_path = os.path.join(run_dir, report["artifacts"]["vector_file"])
    with open(vec_path, "ab") as f:
        f.write(b"tampering_bytes")

    with pytest.raises(RuntimeError, match="Integrity corruption detected"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)

def test_retriever_fails_closed_on_invalid_query_dimensions(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    pipeline.run(str(input_file))

    retriever = CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)

    # Dimension 512 instead of 1024
    with pytest.raises(ValueError, match="does not match index dimension"):
        retriever.retrieve_by_vector([0.1] * 512)

def test_retriever_fails_closed_on_non_finite_query_values(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    pipeline.run(str(input_file))

    retriever = CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)

    nan_vec = [0.1] * 1024
    nan_vec[5] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        retriever.retrieve_by_vector(nan_vec)

def test_retriever_fails_closed_on_invalid_top_k(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    pipeline.run(str(input_file))

    retriever = CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve_by_vector([0.1] * 1024, top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve_by_vector([0.1] * 1024, top_k=-1)

def test_provenance_and_unapproved_disclaimer(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Sample text", "Sample counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    embedder = MockEmbedder(dim=1024)
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    pipeline.run(str(input_file))

    retriever = CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True, embedder=embedder)
    res = retriever.retrieve_by_text("Sample text", top_k=1)

    assert res.scope == "development_provisional"
    assert res.approval_status == "unapproved"
    assert "PROVISIONAL" in res.disclaimer
    assert res.threshold_applied is None  # Reserved for M8
    assert res.provenance["run_id"] is not None
    assert res.provenance["backend"] in ["exact_ip_fallback", "faiss"]

    match = res.matches[0]
    assert match.quality_status == "filtered_candidate"
    assert match.review_status == "unreviewed"

def test_faiss_missing_backend_graceful_fallback():
    # If faiss is not installed, requesting force_backend="faiss" must raise ImportError
    try:
        import faiss
        faiss_available = True
    except ImportError:
        faiss_available = False

    if not faiss_available:
        with pytest.raises(ImportError, match="faiss is required"):
            create_vector_index(dim=1024, force_backend="faiss")

        # But default fallback must succeed and use ExactInnerProductIndex
        index = create_vector_index(dim=1024, prefer_faiss=True)
        assert index.backend_name == "exact_ip_fallback"
        assert index.dimension == 1024
