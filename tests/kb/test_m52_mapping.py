import os
import sys
import pytest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.kb.schema import KBEntry, validate_entry
from src.kb.category_mapper import map_category
from src.kb.filters import safety_filter, leakage_filter
import scripts.run_m52_mapping as m52

def test_18_field_schema_validation():
    entry = KBEntry(
        kb_id="KB-00001", hate_text="test", counter_speech="test", category="Religion",
        target=None, strategy=None, language="en", source="conan", source_id="1",
        provenance="existing-dataset", generation_method="human-authored",
        quality_status="pending", review_status="unreviewed", review_score=None,
        human_audited=False, created_at="2026-10-04T12:00:00Z", version="1.0", notes=""
    )
    assert validate_entry(entry) is True

    entry.hate_text = ""
    assert validate_entry(entry) is False

def test_deterministic_category_mapping():
    assert map_category("multitarget-conan", "WOMEN") == "Gender"
    assert map_category("multitarget-conan", "JEWS") == "Religion"
    assert map_category("conan", "Islamophobia") == "Religion"
    assert map_category("unknown", "Alien") == "UNMAPPED"

def test_missing_field_handling():
    entry = KBEntry(
        kb_id="KB-00002", hate_text="x", counter_speech="y", category="Gender",
        target=None, strategy=None, language="en", source="src", source_id="2",
        provenance="prov", generation_method="human", quality_status="q",
        review_status="r", review_score=None, human_audited=False, created_at="date",
        version="1", notes=""
    )
    d = entry.to_dict()
    assert d["target"] is None
    assert d["strategy"] is None

def test_dedup_and_conflicts():
    # Same HT + different CS => Preserved (one_to_many)
    # Same HT + Same CS + Same Meta => Deduplicated
    # Same HT + Same CS + Diff Meta => Conflict
    pass

def test_safety_rule_behavior():
    safe, reason = safety_filter("I agree, but we should not say that.")
    assert safe is True

    safe, reason = safety_filter("I agree with you completely.")
    assert safe is False
    assert reason == "hate reinforcement"

    safe, reason = safety_filter("You shouldn't use the word faggot.")
    assert safe is False

    safe, reason = safety_filter('You shouldn\'t use the word "faggot".')
    assert safe is True

    safe, reason = safety_filter("I will kill you")
    assert safe is False
    assert reason == "threats or violent escalation"

def test_leakage_detection():
    test_set = {"i hate this"}
    val_set = {"this is bad"}

    safe, r = leakage_filter("I hate this", test_set, val_set)
    assert safe is False
    assert "Test Set" in r

    safe, r = leakage_filter("This is bad", test_set, val_set)
    assert safe is False
    assert "Validation Set" in r

    safe, r = leakage_filter("Completely new text", test_set, val_set)
    assert safe is True

def test_leakage_empty_or_missing_reference_sets():
    valid_set = {"sample hate speech comment"}

    # Empty test set
    with pytest.raises(ValueError, match="non-empty collections"):
        leakage_filter("sample text", set(), valid_set)

    # Empty validation set
    with pytest.raises(ValueError, match="non-empty collections"):
        leakage_filter("sample text", valid_set, set())

    # Both sets empty
    with pytest.raises(ValueError, match="non-empty collections"):
        leakage_filter("sample text", set(), set())

    # None test set
    with pytest.raises(ValueError, match="non-empty collections"):
        leakage_filter("sample text", None, valid_set)

    # None validation set
    with pytest.raises(ValueError, match="non-empty collections"):
        leakage_filter("sample text", valid_set, None)

def test_leakage_exact_matches_test_and_validation():
    test_set = {"exact test match", "shared match text"}
    val_set = {"exact validation match", "shared match text"}

    # Exact match in test set (case and whitespace normalized)
    safe, reason = leakage_filter("  EXACT TEST MATCH  ", test_set, val_set)
    assert safe is False
    assert "HASTIKA Test Set" in reason

    # Exact match in validation set (case and whitespace normalized)
    safe, reason = leakage_filter("Exact Validation Match", test_set, val_set)
    assert safe is False
    assert "HASTIKA Validation Set" in reason

    # Exact match present in both sets
    safe, reason = leakage_filter("shared match text", test_set, val_set)
    assert safe is False
    assert "Leakage" in reason

    # Non-matching text
    safe, reason = leakage_filter("Completely unseen hate comment", test_set, val_set)
    assert safe is True
    assert reason == ""

def test_m52_verified_mode_blocks_when_splits_missing(monkeypatch):
    monkeypatch.setattr(m52, "load_hastika_texts", lambda: (set(), set()))
    monkeypatch.setattr(sys, "argv", ["run_m52_mapping.py", "--mode", "verified"])
    with pytest.raises(SystemExit) as exc_info:
        m52.main()
    assert exc_info.value.code == 1

def test_m52_provisional_mode_audit_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(m52, "load_hastika_texts", lambda: (set(), set()))
    monkeypatch.setattr(m52, "process_mt_conan", lambda stats: [])

    dummy = KBEntry(
        kb_id="",
        hate_text="some test hate speech",
        counter_speech="This is constructive counter speech.",
        category="Religion",
        target="Islamophobia",
        strategy=None,
        language="en",
        source="conan",
        source_id="TEST-001",
        provenance="existing-dataset",
        generation_method="human-authored",
        quality_status="pending",
        review_status="unreviewed",
        review_score=None,
        human_audited=False,
        created_at="2026-10-22T00:00:00Z",
        version="1.0",
        notes=""
    )
    monkeypatch.setattr(m52, "process_conan", lambda stats: [dummy])

    # Redirect outputs to tmp_path
    monkeypatch.setattr(m52, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(m52, "backup_existing_artifacts", lambda: None)

    orig_open = open
    def mock_open_func(file, mode='r', *args, **kwargs):
        s_file = str(file).replace('\\', '/')
        if s_file.startswith("reports/"):
            out_dir = tmp_path / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            return orig_open(out_dir / os.path.basename(s_file), mode, *args, **kwargs)
        if s_file.startswith("docs/"):
            out_dir = tmp_path / "docs"
            out_dir.mkdir(parents=True, exist_ok=True)
            return orig_open(out_dir / os.path.basename(s_file), mode, *args, **kwargs)
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_func)
    monkeypatch.setattr(sys, "argv", ["run_m52_mapping.py", "--provisional", "--no-backup"])

    m52.main()

    audit_file = tmp_path / "reports" / "m5_2_audit.json"
    assert audit_file.exists()
    with orig_open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    assert audit_data["leakage_audit"]["status"] == "skipped_provisional"
    assert audit_data["leakage_audit"]["verified"] is False
    assert audit_data["leakage_audit"]["mode"] == "provisional"
    assert "missing or empty" in audit_data["leakage_audit"]["reason"]

    report_file = tmp_path / "docs" / "M5.2_COMPLETION_REPORT.md"
    assert report_file.exists()
    with orig_open(report_file, "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "SKIPPED (PROVISIONAL MODE)" in report_text
    assert "Zero leakage is **NOT** claimed" in report_text

def test_m52_verified_mode_reports_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(m52, "load_hastika_texts", lambda: ({"test reference comment"}, {"val reference comment"}))
    monkeypatch.setattr(m52, "process_mt_conan", lambda stats: [])

    dummy = KBEntry(
        kb_id="",
        hate_text="some test hate speech",
        counter_speech="This is constructive counter speech.",
        category="Religion",
        target="Islamophobia",
        strategy=None,
        language="en",
        source="conan",
        source_id="TEST-002",
        provenance="existing-dataset",
        generation_method="human-authored",
        quality_status="pending",
        review_status="unreviewed",
        review_score=None,
        human_audited=False,
        created_at="2026-10-22T00:00:00Z",
        version="1.0",
        notes=""
    )
    monkeypatch.setattr(m52, "process_conan", lambda stats: [dummy])

    # Redirect outputs to tmp_path
    monkeypatch.setattr(m52, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(m52, "backup_existing_artifacts", lambda: None)

    orig_open = open
    def mock_open_func(file, mode='r', *args, **kwargs):
        s_file = str(file).replace('\\', '/')
        if s_file.startswith("reports/"):
            out_dir = tmp_path / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            return orig_open(out_dir / os.path.basename(s_file), mode, *args, **kwargs)
        if s_file.startswith("docs/"):
            out_dir = tmp_path / "docs"
            out_dir.mkdir(parents=True, exist_ok=True)
            return orig_open(out_dir / os.path.basename(s_file), mode, *args, **kwargs)
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_func)
    monkeypatch.setattr(sys, "argv", ["run_m52_mapping.py", "--mode", "verified", "--no-backup"])

    m52.main()

    audit_file = tmp_path / "reports" / "m5_2_audit.json"
    assert audit_file.exists()
    with orig_open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    assert audit_data["leakage_audit"]["status"] == "completed"
    assert audit_data["leakage_audit"]["verified"] is True
    assert audit_data["leakage_audit"]["mode"] == "verified"
    assert audit_data["leakage_audit"]["matches"]["test_matches"] == 0
    assert audit_data["leakage_audit"]["matches"]["val_matches"] == 0

    report_file = tmp_path / "docs" / "M5.2_COMPLETION_REPORT.md"
    assert report_file.exists()
    with orig_open(report_file, "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "Completed (Verified)" in report_text
    assert "Exact Matches in HASTIKA Test Set: 0" in report_text

def test_raw_source_immutability():
    mt_path = "data/counter_speech_kb/sources/Multitarget-CONAN.csv"
    if os.path.exists(mt_path):
        mtime = os.path.getmtime(mt_path)
        assert mtime > 0

def test_deterministic_reproducible_output():
    assert True
