import os
import sys
import json
import math
import hashlib
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
    assert not os.path.exists(os.path.join(str(tmp_path / "reports"), "m6_skipped_records.jsonl.tmp"))

def test_skipped_records_atomic_replacement_success(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    os.makedirs(rep_dir, exist_ok=True)

    # Pre-populate a prior skipped records artifact
    prior_skipped_path = os.path.join(rep_dir, "m6_skipped_records.jsonl")
    with open(prior_skipped_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kb_id": "OLD-001", "reason": "Old skip"}) + "\n")

    input_file = tmp_path / "input.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Valid text", "Counter"),
        create_sample_kb_entry("KB-00002", "   ", "Counter"),  # Triggers skip
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    report = pipeline.run(str(input_file))

    assert report["skipped_record_count"] == 1
    assert os.path.exists(prior_skipped_path)
    assert not os.path.exists(prior_skipped_path + ".tmp")

    with open(prior_skipped_path, "r", encoding="utf-8") as f:
        content = [json.loads(line) for line in f]
    assert len(content) == 1
    assert content[0]["kb_id"] == "KB-00002"

    # Verify prior file was safely backed up
    backup_base = os.path.join(rep_dir, "backups")
    assert os.path.exists(backup_base)
    backup_dirs = os.listdir(backup_base)
    assert len(backup_dirs) >= 1
    backup_skipped = os.path.join(backup_base, backup_dirs[0], "m6_skipped_records.jsonl")
    assert os.path.exists(backup_skipped)
    with open(backup_skipped, "r", encoding="utf-8") as f:
        backup_content = json.loads(f.readline())
        assert backup_content["kb_id"] == "OLD-001"

def test_skipped_records_preservation_and_tmp_cleanup_on_pipeline_failure(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    os.makedirs(rep_dir, exist_ok=True)

    prior_skipped_path = os.path.join(rep_dir, "m6_skipped_records.jsonl")
    with open(prior_skipped_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kb_id": "PRESERVE-001", "reason": "Initial valid skip"}) + "\n")

    input_file = tmp_path / "input.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Valid text", "Counter"),
        create_sample_kb_entry("KB-00002", "   ", "Counter"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    class FaultyEmbedder(BaseEmbedder):
        @property
        def dimension(self): return 1024
        @property
        def model_identifier(self): return "FAULTY"
        @property
        def is_mock(self): return True
        def embed_batch(self, texts):
            raise RuntimeError("Crash during inference")

    pipeline = EmbeddingPipeline(
        embedder=FaultyEmbedder(),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )

    with pytest.raises(RuntimeError, match="Crash during inference"):
        pipeline.run(str(input_file))

    # Staging temp must not exist, prior file must be untouched
    assert not os.path.exists(prior_skipped_path + ".tmp")
    assert os.path.exists(prior_skipped_path)
    with open(prior_skipped_path, "r", encoding="utf-8") as f:
        preserved = json.loads(f.readline())
        assert preserved["kb_id"] == "PRESERVE-001"

def test_skipped_records_preservation_on_write_failure(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    os.makedirs(rep_dir, exist_ok=True)

    prior_skipped_path = os.path.join(rep_dir, "m6_skipped_records.jsonl")
    with open(prior_skipped_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kb_id": "PRESERVE-WRITE-FAIL", "reason": "Initial skip"}) + "\n")

    input_file = tmp_path / "input.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Valid text", "Counter"),
        create_sample_kb_entry("KB-00002", "   ", "Counter"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    # Simulate failure in json.dump when writing the report (after tmp_skipped_path was written)
    def broken_dump(*args, **kwargs):
        raise OSError("Simulated disk error during report dump")
    monkeypatch.setattr(json, "dump", broken_dump)

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )

    with pytest.raises(RuntimeError, match="Simulated disk error"):
        pipeline.run(str(input_file))

    # Verify tmp file was deleted in cleanup and prior file was preserved
    assert not os.path.exists(prior_skipped_path + ".tmp")
    assert os.path.exists(prior_skipped_path)
    with open(prior_skipped_path, "r", encoding="utf-8") as f:
        preserved = json.loads(f.readline())
        assert preserved["kb_id"] == "PRESERVE-WRITE-FAIL"

def test_skipped_records_cleanup_when_subsequent_run_has_zero_skips(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    os.makedirs(rep_dir, exist_ok=True)

    prior_skipped_path = os.path.join(rep_dir, "m6_skipped_records.jsonl")
    with open(prior_skipped_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kb_id": "OLD-SKIP", "reason": "Old skip"}) + "\n")

    input_file = tmp_path / "input.jsonl"
    entries = [
        create_sample_kb_entry("KB-00001", "Valid text 1", "Counter 1"),
        create_sample_kb_entry("KB-00002", "Valid text 2", "Counter 2"),
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="development_provisional",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    report = pipeline.run(str(input_file))

    assert report["skipped_record_count"] == 0
    # The stale skipped file must be removed from the active reports directory
    assert not os.path.exists(prior_skipped_path)

    # But it must be safely archived in the pre-run backup
    backup_base = os.path.join(rep_dir, "backups")
    assert os.path.exists(backup_base)
    backup_dirs = os.listdir(backup_base)
    assert len(backup_dirs) >= 1
    backup_skipped = os.path.join(backup_base, backup_dirs[0], "m6_skipped_records.jsonl")
    assert os.path.exists(backup_skipped)

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

def test_failure_during_artifact_write_leaves_previous_run_selectable(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    # Run 1 succeeds and commits
    p1 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    r1 = p1.run(str(input_file))
    resolved_1 = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved_1["run_id"] == r1["run_id"]

    # Run 2 encounters failure while writing vector matrix
    import src.retrieval.embedder as emb_mod
    orig_save = emb_mod.save_npy_f32
    def failing_save(*args, **kwargs):
        raise OSError("Simulated disk error during vector write")
    monkeypatch.setattr(emb_mod, "save_npy_f32", failing_save)

    p2 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    with pytest.raises(RuntimeError, match="Simulated disk error"):
        p2.run(str(input_file))

    # Verify previous committed run is still authoritative and selectable
    resolved_after = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved_after["run_id"] == resolved_1["run_id"]

def test_failure_during_validation_leaves_previous_run_selectable(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    p1 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    r1 = p1.run(str(input_file))
    resolved_1 = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)

    class NanEmbedder(BaseEmbedder):
        @property
        def dimension(self): return 1024
        @property
        def model_identifier(self): return "NAN-MOCK"
        @property
        def is_mock(self): return True
        def embed_batch(self, texts):
            vec = [float("nan")] * 1024
            return [vec for _ in texts]

    p2 = EmbeddingPipeline(embedder=NanEmbedder(), output_dir=out_dir, reports_dir=rep_dir)
    with pytest.raises(RuntimeError, match="Non-finite value"):
        p2.run(str(input_file))

    resolved_after = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved_after["run_id"] == resolved_1["run_id"]

def test_failure_during_marker_creation_leaves_previous_run_selectable(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    p1 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    r1 = p1.run(str(input_file))
    resolved_1 = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)

    orig_open = open
    def failing_open(file, *args, **kwargs):
        if "COMMITTED.tmp" in str(file):
            raise OSError("Simulated disk error during commit marker write")
        return orig_open(file, *args, **kwargs)
    monkeypatch.setattr("builtins.open", failing_open)

    p2 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    with pytest.raises(RuntimeError, match="Simulated disk error during commit marker write"):
        p2.run(str(input_file))

    resolved_after = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved_after["run_id"] == resolved_1["run_id"]

def test_failure_during_pointer_update_leaves_previous_run_selectable(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    p1 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    r1 = p1.run(str(input_file))
    resolved_1 = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)

    orig_replace = os.replace
    def failing_replace(src, dst):
        if "LATEST.json" in str(dst):
            raise OSError("Simulated lock on LATEST.json")
        return orig_replace(src, dst)
    monkeypatch.setattr(os, "replace", failing_replace)

    p2 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    with pytest.raises(RuntimeError, match="Simulated lock on LATEST.json"):
        p2.run(str(input_file))

    resolved_after = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved_after["run_id"] == resolved_1["run_id"]

def test_failed_backup_aborts_immediately_without_modifying_active_artifacts(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    p1 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    r1 = p1.run(str(input_file))
    resolved_1 = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)

    import shutil
    orig_copy2 = shutil.copy2
    def failing_copy(src, dst):
        if ".tmp" in str(dst) and "backups" in str(dst):
            raise OSError("Simulated disk error during backup")
        return orig_copy2(src, dst)
    monkeypatch.setattr(shutil, "copy2", failing_copy)

    p2 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    with pytest.raises(RuntimeError, match="Pre-run backup failed and aborted safely"):
        p2.run(str(input_file))

    # Verify no broken backup folder was left as complete
    backup_base = os.path.join(rep_dir, "backups")
    for b_item in os.listdir(backup_base):
        b_path = os.path.join(backup_base, b_item)
        if os.path.isdir(b_path):
            assert not b_item.endswith(".tmp")

    resolved_after = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved_after["run_id"] == resolved_1["run_id"]

def test_simulated_process_interruption_leaves_run_uncommitted(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    runs_dir = os.path.join(out_dir, "runs")
    interrupted_run_dir = os.path.join(runs_dir, "bge_m3_mock_interrupted_run")
    os.makedirs(interrupted_run_dir, exist_ok=True)

    # Put files in run directory without COMMITTED marker
    fake_vec = os.path.join(interrupted_run_dir, "bge_m3_mock_dense.npy")
    save_npy_f32(fake_vec, [[0.1] * 1024], (1, 1024))

    # Point LATEST.json to the interrupted run
    pointer_path = os.path.join(out_dir, "LATEST.json")
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": "interrupted", "run_dir": "runs/bge_m3_mock_interrupted_run"}, f)

    with pytest.raises(RuntimeError, match="uncommitted"):
        EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)

def test_m7_resolver_fails_closed_on_mock_in_approved_scope(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    pipeline.run(str(input_file))

    # Fails closed if allow_mock=False
    with pytest.raises(ValueError, match="Refusing to consume mock embeddings"):
        EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=False)

    # Succeeds if allow_mock=True
    resolved = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    assert resolved["is_mock"] is True

def test_m7_resolver_fails_closed_on_scope_mismatch(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
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
        EmbeddingPipeline.resolve_latest_run(out_dir, expected_scope="approved_kb", allow_mock=True)

    resolved = EmbeddingPipeline.resolve_latest_run(out_dir, expected_scope="development_provisional", allow_mock=True)
    assert resolved["scope"] == "development_provisional"

def test_m7_resolver_detects_cryptographic_corruption(tmp_path):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    pipeline.run(str(input_file))

    resolved = EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)
    # Corrupt vector file on disk
    with open(resolved["vector_file"], "ab") as f:
        f.write(b"corruption")

    with pytest.raises(RuntimeError, match="Integrity corruption detected"):
        EmbeddingPipeline.resolve_latest_run(out_dir, allow_mock=True)

def test_canonical_mirror_failure_triggers_rollback_from_backup(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    input_file = tmp_path / "candidates.jsonl"
    entries = [create_sample_kb_entry("KB-00001", "Text 1", "Counter 1")]
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entries[0].to_dict()) + "\n")

    # Run 1 establishes canonical files
    p1 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    r1 = p1.run(str(input_file))
    canonical_vec = os.path.join(out_dir, "bge_m3_mock_dense.npy")
    assert os.path.exists(canonical_vec)
    with open(canonical_vec, "rb") as f:
        orig_canonical_hash = hashlib.sha256(f.read()).hexdigest()

    # Run 2 fails once during canonical mirror replacement, then rollback succeeds
    call_count = 0
    orig_replace = os.replace
    def failing_replace(src, dst):
        nonlocal call_count
        if "bge_m3_mock_dense.npy" in str(dst) and "runs" not in str(dst) and call_count == 0:
            call_count += 1
            raise OSError("Simulated lock on canonical vector mirror")
        return orig_replace(src, dst)
    monkeypatch.setattr(os, "replace", failing_replace)

    p2 = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), output_dir=out_dir, reports_dir=rep_dir)
    with pytest.raises(RuntimeError, match="Canonical mirror publication failed; state rolled back"):
        p2.run(str(input_file))

    # Verify canonical vector file was restored from backup and matches Run 1
    assert os.path.exists(canonical_vec)
    with open(canonical_vec, "rb") as f:
        restored_hash = hashlib.sha256(f.read()).hexdigest()
    assert restored_hash == orig_canonical_hash
