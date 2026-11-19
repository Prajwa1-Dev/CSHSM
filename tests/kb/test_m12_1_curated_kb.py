"""
Unit and integration tests for Milestone 12.1: Curated KB and Disk Retrieval.

Verifies:
1. Review queue schema and pending status integrity.
2. Approved KB fail-closed behavior when unreviewed/pending candidates are supplied.
3. Approved KB success when valid human-approved records are supplied.
4. Successful loading of committed mock embedding run from disk via CSHSMRetriever.
5. Vector and manifest row alignment, dimension matching, and provenance preservation.
6. Fail-closed error handling when LATEST.json, COMMITTED marker, or files are corrupt/missing.
"""

import os
import sys
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.kb.schema import KBEntry, validate_entry
from src.retrieval.embedder import (
    BaseEmbedder,
    MockEmbedder,
    EmbeddingPipeline,
    load_npy_f32
)
from src.retrieval.retriever import (
    CSHSMRetriever,
    ExactInnerProductIndex
)


def create_sample_entry(
    kb_id: str = "KB-TEST-01",
    hate_text: str = "Sample discriminatory remark against a group.",
    counter_speech: str = "Every individual deserves equal dignity and protection under the law.",
    category: str = "Religion",
    target: str = "Islamophobia",
    quality_status: str = "pending_review",
    review_status: str = "pending_review",
    human_audited: bool = False
) -> KBEntry:
    return KBEntry(
        kb_id=kb_id,
        hate_text=hate_text,
        counter_speech=counter_speech,
        category=category,
        target=target,
        strategy="reframing",
        language="en",
        source="conan",
        source_id="101",
        provenance="existing-dataset",
        generation_method="human-authored",
        quality_status=quality_status,
        review_status=review_status,
        review_score=None,
        human_audited=human_audited,
        created_at="2026-10-24T00:00:00Z",
        version="1.0",
        notes="M12.1 test fixture"
    )


# =====================================================================
# Tests: Review Queue & Schema Invariants
# =====================================================================

def test_review_queue_file_exists_and_conforms_to_schema():
    """Review queue must exist, have 25-30 records, and strictly preserve unreviewed status."""
    queue_path = "data/counter_speech_kb/review_queue.jsonl"
    assert os.path.exists(queue_path), f"Review queue missing at: {queue_path}"

    records = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert 25 <= len(records) <= 30, f"Expected 25-30 records, found {len(records)}"

    seen_ids = set()
    seen_hate = set()
    seen_counter = set()
    categories = set()

    for r in records:
        entry = KBEntry(**r)
        assert validate_entry(entry), f"Record {entry.kb_id} failed KBEntry validation."

        # Strict unapproved invariant: NO automated approval
        assert entry.human_audited is False
        assert entry.quality_status == "pending_review"
        assert entry.review_status == "pending_review"
        assert entry.review_score is None

        # Diversity & Uniqueness
        assert entry.kb_id not in seen_ids, f"Duplicate kb_id: {entry.kb_id}"
        assert entry.hate_text not in seen_hate, f"Duplicate hate_text: {entry.hate_text}"
        assert entry.counter_speech not in seen_counter, f"Duplicate counter_speech: {entry.counter_speech}"

        seen_ids.add(entry.kb_id)
        seen_hate.add(entry.hate_text)
        seen_counter.add(entry.counter_speech)
        categories.add(entry.category)

    # Must cover core categories
    assert {"Religion", "Gender", "Geo-political", "Others"}.issubset(categories)


# =====================================================================
# Tests: Approved Scope Fail-Closed Behavior
# =====================================================================

def test_approved_kb_embedding_fails_closed_on_unreviewed_candidates(tmp_path):
    """EmbeddingPipeline(scope='approved_kb') must reject records not human-approved."""
    input_file = tmp_path / "test_unreviewed.jsonl"
    unreviewed = create_sample_entry(
        quality_status="pending_review",
        review_status="pending_review",
        human_audited=False
    )
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(unreviewed.to_dict()) + "\n")

    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="approved_kb",
        output_dir=str(tmp_path / "embeddings"),
        reports_dir=str(tmp_path / "reports")
    )

    with pytest.raises(ValueError, match="Approved KB mode fail-closed"):
        pipeline.run(str(input_file))


def test_approved_kb_embedding_succeeds_with_human_approved_candidates(tmp_path):
    """EmbeddingPipeline(scope='approved_kb') succeeds when records are human-approved."""
    input_file = tmp_path / "test_approved.jsonl"
    approved_1 = create_sample_entry(
        kb_id="KB-APP-01",
        quality_status="approved",
        review_status="human-approved",
        human_audited=True
    )
    approved_2 = create_sample_entry(
        kb_id="KB-APP-02",
        hate_text="Second distinct hateful comment.",
        counter_speech="Second distinct constructive counter response.",
        quality_status="approved",
        review_status="human-approved",
        human_audited=True
    )
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(approved_1.to_dict()) + "\n")
        f.write(json.dumps(approved_2.to_dict()) + "\n")

    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dim=1024),
        scope="approved_kb",
        output_dir=out_dir,
        reports_dir=rep_dir
    )

    report = pipeline.run(str(input_file))
    assert report["kb_scope"] == "approved_kb"
    assert report["approval_status"] == "approved"
    assert report["embedded_record_count"] == 2
    assert report["publication_status"] == "committed"

    # Verify LATEST.json pointer was created and committed
    pointer_path = os.path.join(out_dir, "LATEST.json")
    assert os.path.exists(pointer_path)
    with open(pointer_path, "r", encoding="utf-8") as f:
        pointer = json.load(f)
    assert pointer["scope"] == "approved_kb"
    assert pointer["manifest_count"] == 2


# =====================================================================
# Tests: Disk-Based Retrieval Loading & Row Alignment
# =====================================================================

def test_disk_retrieval_loads_committed_approved_run(tmp_path):
    """CSHSMRetriever.from_committed_run loads committed approved artifacts from disk."""
    input_file = tmp_path / "curated_approved.jsonl"
    entries = [
        create_sample_entry(f"KB-C-{i:02d}", f"Hate query text {i}", f"Counter narrative {i}", quality_status="approved", review_status="human-approved", human_audited=True)
        for i in range(5)
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")

    out_dir = str(tmp_path / "embeddings")
    rep_dir = str(tmp_path / "reports")
    embedder = MockEmbedder(dim=1024)

    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="approved_kb",
        output_dir=out_dir,
        reports_dir=rep_dir
    )
    pipeline.run(str(input_file))

    # Load via authoritative from_committed_run
    retriever = CSHSMRetriever.from_committed_run(
        output_dir=out_dir,
        scope="approved_kb",
        allow_mock=True,
        embedder=embedder
    )

    assert retriever.index.ntotal == 5
    assert len(retriever.manifest) == 5
    assert retriever.run_metadata["scope"] == "approved_kb"
    assert retriever.run_metadata["is_mock"] is True

    # Row alignment check
    for idx, row in enumerate(retriever.manifest):
        assert row["row_idx"] == idx
        assert row["kb_id"] == f"KB-C-{idx:02d}"

    # Perform actual retrieval from disk
    result = retriever.retrieve_by_text("Hate query text 2", top_k=2)
    assert result.retrieval_status == "success"
    assert len(result.matches) == 2
    assert result.matches[0].kb_id == "KB-C-02"
    assert result.matches[0].counter_speech == "Counter narrative 2"
    assert result.matches[0].quality_status == "approved"


# =====================================================================
# Tests: Fail-Closed Corruption & Tampering Protections
# =====================================================================

def test_disk_resolver_fails_closed_when_pointer_missing(tmp_path):
    """Resolver fails closed if LATEST.json is absent."""
    out_dir = str(tmp_path / "empty_dir")
    os.makedirs(out_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="No authoritative run pointer"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)


def test_disk_resolver_fails_closed_when_commit_marker_missing(tmp_path):
    """Resolver fails closed if COMMITTED seal is removed from the run directory."""
    input_file = tmp_path / "test.jsonl"
    entry = create_sample_entry(quality_status="approved", review_status="human-approved", human_audited=True)
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")

    out_dir = str(tmp_path / "embeddings")
    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), scope="approved_kb", output_dir=out_dir)
    pipeline.run(str(input_file))

    # Read pointer to find run_dir
    with open(os.path.join(out_dir, "LATEST.json"), "r", encoding="utf-8") as f:
        pointer = json.load(f)
    run_dir = os.path.join(out_dir, pointer["run_dir"])

    # Remove COMMITTED marker
    os.remove(os.path.join(run_dir, "COMMITTED"))

    with pytest.raises(RuntimeError, match="is uncommitted"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)


def test_disk_resolver_fails_closed_when_vector_checksum_tampered(tmp_path):
    """Resolver fails closed if dense vector file is modified on disk."""
    input_file = tmp_path / "test.jsonl"
    entry = create_sample_entry(quality_status="approved", review_status="human-approved", human_audited=True)
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")

    out_dir = str(tmp_path / "embeddings")
    pipeline = EmbeddingPipeline(embedder=MockEmbedder(dim=1024), scope="approved_kb", output_dir=out_dir)
    pipeline.run(str(input_file))

    with open(os.path.join(out_dir, "LATEST.json"), "r", encoding="utf-8") as f:
        pointer = json.load(f)
    run_dir = os.path.join(out_dir, pointer["run_dir"])

    # Tamper with vector file
    vec_path = os.path.join(run_dir, "bge_m3_mock_dense.npy")
    with open(vec_path, "ab") as f:
        f.write(b"\x00\x00\x00\x00")

    with pytest.raises(RuntimeError, match="vector file SHA-256 mismatch"):
        CSHSMRetriever.from_committed_run(output_dir=out_dir, allow_mock=True)


# =====================================================================
# Tests: AI Quality Review Report Integrity & Non-Approval Invariants
# =====================================================================

def test_ai_quality_review_report_integrity():
    """Verifies AI quality review report schema, completeness, and non-approval invariants."""
    report_path = "reports/m12_1_ai_quality_review.json"
    queue_path = "data/counter_speech_kb/review_queue.jsonl"

    assert os.path.exists(report_path), f"AI quality review report missing at: {report_path}"
    assert os.path.exists(queue_path), f"Review queue missing at: {queue_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Top-level non-approval invariants
    assert report["review_method"] == "ai_assisted"
    assert report["human_approved"] is False
    assert "NOT constitute verified human approval" in report["disclaimer"]

    # Load expected candidate IDs from the queue
    expected_ids = set()
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                expected_ids.add(json.loads(line)["kb_id"])

    assessments = report.get("candidate_assessments", [])
    assert len(assessments) == 28, f"Expected 28 assessments, found {len(assessments)}"

    seen_ids = set()
    recommendation_counts = {"acceptable_with_caveats": 0, "revise": 0, "reject": 0}
    allowed_recommendations = set(recommendation_counts.keys())

    for item in assessments:
        kb_id = item.get("kb_id")
        assert kb_id is not None, "Assessment item missing kb_id"
        assert kb_id not in seen_ids, f"Duplicate kb_id in assessments: {kb_id}"
        assert kb_id in expected_ids, f"Unknown kb_id in assessments: {kb_id}"
        seen_ids.add(kb_id)

        rec = item.get("recommendation")
        assert rec in allowed_recommendations, f"Invalid recommendation '{rec}' for {kb_id}"
        recommendation_counts[rec] += 1

        assert item.get("reason"), f"Missing reason for {kb_id}"
        assert item.get("caveats") is not None, f"Missing caveats for {kb_id}"
        assert item.get("review_method") == "ai_assisted", f"Invalid review_method for {kb_id}"
        assert item.get("human_approved") is False, f"human_approved must be False for {kb_id}"

    # Verify all expected IDs from the queue were assessed exactly once
    assert seen_ids == expected_ids, "Mismatch between queue candidate IDs and assessment IDs"

    # Verify summary counts match assessment records
    summary = report.get("summary", {})
    assert summary.get("total_reviewed") == 28
    assert summary.get("acceptable_with_caveats") == recommendation_counts["acceptable_with_caveats"]
    assert summary.get("revise") == recommendation_counts["revise"]
    assert summary.get("reject") == recommendation_counts["reject"]
    assert summary.get("human_approved_count") == 0


def test_review_queue_and_candidate_data_remain_unmodified():
    """Verifies that the original review queue and candidate datasets were not altered or approved."""
    queue_path = "data/counter_speech_kb/review_queue.jsonl"
    filtered_path = "data/counter_speech_kb/filtered_candidates.jsonl"

    # 1. Review queue must still contain 28 records, all strictly unapproved
    with open(queue_path, "r", encoding="utf-8") as f:
        queue_records = [json.loads(l) for l in f if l.strip()]

    assert len(queue_records) == 28
    for r in queue_records:
        assert r["human_audited"] is False
        assert r["quality_status"] == "pending_review"
        assert r["review_status"] == "pending_review"
        assert r["review_score"] is None

    # 2. Source candidates must remain unreviewed (19,436 records)
    with open(filtered_path, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    assert count == 19436

    # 3. No approved_kb.jsonl should have been created from AI assessment alone
    approved_path = "data/counter_speech_kb/approved_kb.jsonl"
    assert not os.path.exists(approved_path), (
        "approved_kb.jsonl must not be created based solely on AI quality review."
    )
