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
    BGE_M3_Embedder,
    EmbeddingPipeline,
    save_npy_f32,
    load_npy_f32
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

def test_mock_embedder_dimension_and_normalization():
    embedder = MockEmbedder(dim=1024)
    assert embedder.dimension == 1024
    assert embedder.is_mock is True
    assert embedder.model_identifier == "MOCK-EMBEDDER-TEST-ONLY"

    texts = ["First comment", "Second distinct comment", "First comment"]
    vecs = embedder.embed_batch(texts)

    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 1024
        # Assert finite
        assert all(not (math.isnan(x) or math.isinf(x)) for x in v)
        # Assert L2 unit norm
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-4

    # Assert deterministic mapping
    assert vecs[0] == vecs[2]
    assert vecs[0] != vecs[1]

def test_pure_python_npy_save_and_load(tmp_path):
    fpath = str(tmp_path / "test_matrix.npy")
    test_data = [
        [1.0, 0.0, 0.5, -0.5],
        [0.0, 1.0, -0.5, 0.5],
        [0.5, -0.5, 1.0, 0.0]
    ]
    shape = (3, 4)
    save_npy_f32(fpath, test_data, shape)

    assert os.path.exists(fpath)
    loaded_data, loaded_shape = load_npy_f32(fpath)

    assert loaded_shape == shape
    assert len(loaded_data) == 3
    for r_orig, r_load in zip(test_data, loaded_data):
        for v_orig, v_load in zip(r_orig, r_load):
            assert abs(v_orig - v_load) < 1e-6

def test_pipeline_row_to_manifest_alignment(tmp_path):
    input_file = tmp_path / "candidates.jsonl"
    entries = [
        create_sample_kb_entry(f"KB-{i+1:05d}", f"Hate text {i+1}", f"Counter {i+1}")
        for i in range(10)
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")

    embedder = MockEmbedder(dim=1024)
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="development_provisional",
        batch_size=3,
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    report = pipeline.run(str(input_file))

    vec_path = os.path.join(out_dir, "bge_m3_mock_dense.npy")
    manifest_path = os.path.join(out_dir, "bge_m3_mock_manifest.jsonl")

    assert os.path.exists(vec_path)
    assert os.path.exists(manifest_path)

    vecs, shape = load_npy_f32(vec_path)
    assert shape == (10, 1024)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_lines = [json.loads(line) for line in f]

    assert len(manifest_lines) == 10
    for idx, (m_row, v) in enumerate(zip(manifest_lines, vecs)):
        assert m_row["row_idx"] == idx
        assert m_row["kb_id"] == f"KB-{idx+1:05d}"
        assert m_row["scope"] == "development_provisional"
        assert m_row["quality_status"] == "filtered_candidate"
        assert len(v) == 1024
        # Verify L2 norm
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-4

def test_pipeline_development_provisional_scope_isolation(tmp_path):
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Hate text", "Counter", quality_status="filtered_candidate")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    report = pipeline.run(str(input_file))

    assert report["kb_scope"] == "development_provisional"
    assert report["approval_status"] == "unapproved"
    assert "PROVISIONAL" in report["disclaimer"]
    assert report["mock_notice"] is not None

def test_pipeline_approved_kb_fails_closed_on_unapproved_candidates(tmp_path):
    input_file = tmp_path / "unapproved_candidates.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Hate text", "Counter", quality_status="filtered_candidate", review_status="unreviewed")
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="approved_kb",
        output_dir=str(tmp_path / "embeddings"),
        reports_dir=str(tmp_path / "reports")
    )

    # Must fail closed immediately
    with pytest.raises(ValueError, match="Approved KB mode fail-closed"):
        pipeline.run(str(input_file))

def test_pipeline_approved_kb_fails_closed_on_empty_or_missing_file(tmp_path):
    missing_path = str(tmp_path / "non_existent.jsonl")
    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="approved_kb",
        output_dir=str(tmp_path / "embeddings"),
        reports_dir=str(tmp_path / "reports")
    )
    with pytest.raises(FileNotFoundError):
        pipeline.run(missing_path)

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("")
    with pytest.raises(ValueError, match="fail-closed"):
        pipeline.run(str(empty_path))

def test_pipeline_approved_kb_succeeds_only_with_genuinely_approved_records(tmp_path):
    input_file = tmp_path / "approved_kb.jsonl"
    approved_entry = create_sample_kb_entry(
        "KB-00001", "Hate text", "Counter",
        quality_status="approved",
        review_status="approved"
    )
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(approved_entry.to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="approved_kb",
        output_dir=str(tmp_path / "embeddings"),
        reports_dir=str(tmp_path / "reports")
    )
    report = pipeline.run(str(input_file))
    assert report["kb_scope"] == "approved_kb"
    assert report["approval_status"] == "approved"

def test_pipeline_empty_and_malformed_text_handling(tmp_path):
    input_file = tmp_path / "mixed_candidates.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Valid text", "Counter"),
        create_sample_kb_entry("KB-00002", "   ", "Counter"), # Empty hate_text
        create_sample_kb_entry("KB-00003", "Another valid text", "Counter"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    rep_dir = str(tmp_path / "reports")
    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=str(tmp_path / "embeddings"),
        reports_dir=rep_dir
    )
    report = pipeline.run(str(input_file))

    assert report["raw_record_count"] == 3
    assert report["skipped_record_count"] == 1
    assert report["embedded_record_count"] == 2
    assert report["matrix_shape"] == [2, 1024]

    skipped_file = os.path.join(rep_dir, "m6_skipped_records.jsonl")
    assert os.path.exists(skipped_file)
    with open(skipped_file, "r", encoding="utf-8") as f:
        skipped = json.loads(f.readline())
        assert skipped["kb_id"] == "KB-00002"
        assert "Empty" in skipped["reason"]

def test_pipeline_atomic_rollback_on_failure(tmp_path):
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text", "Counter")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    class FaultyEmbedder(BaseEmbedder):
        @property
        def dimension(self): return 1024
        @property
        def model_identifier(self): return "FAULTY"
        @property
        def is_mock(self): return True
        def embed_batch(self, texts):
            raise RuntimeError("Simulated OOM crash during forward pass")

    out_dir = str(tmp_path / "embeddings")
    pipeline = EmbeddingPipeline(
        embedder=FaultyEmbedder(),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=str(tmp_path / "reports")
    )
    with pytest.raises(RuntimeError, match="Simulated OOM crash"):
        pipeline.run(str(input_file))

    # Assert no partial .tmp files left behind
    assert not os.path.exists(os.path.join(out_dir, "bge_m3_mock_dense.npy"))
    assert not os.path.exists(os.path.join(out_dir, "bge_m3_mock_dense.npy.tmp"))

def test_mock_vs_real_embedder_guard():
    mock_emb = MockEmbedder(dim=1024)
    assert mock_emb.is_mock is True
    assert "MOCK" in mock_emb.model_identifier

    real_emb = BGE_M3_Embedder(model_name="BAAI/bge-m3")
    assert real_emb.is_mock is False
    assert real_emb.model_identifier == "BAAI/bge-m3"

    # In the current environment where sentence-transformers is not installed,
    # attempting to embed must raise ImportError and not silently generate mock vectors
    with pytest.raises(ImportError, match="sentence-transformers is required"):
        real_emb.embed_batch(["test text"])

def test_dev_limit_100_subset(tmp_path):
    input_file = tmp_path / "many_candidates.jsonl"
    entries = [
        create_sample_kb_entry(f"KB-{i+1:05d}", f"Hate text {i+1}", f"Counter {i+1}")
        for i in range(150)
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        dev_limit=100,
        output_dir=str(tmp_path / "embeddings"),
        reports_dir=str(tmp_path / "reports")
    )
    report = pipeline.run(str(input_file))
    assert report["raw_record_count"] == 150
    assert report["embedded_record_count"] == 100
    assert report["matrix_shape"] == [100, 1024]
