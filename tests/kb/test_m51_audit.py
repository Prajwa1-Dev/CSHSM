import os
import pytest
import json

REPORTS_DIR = "reports"
DATA_DIR = "data/counter_speech_kb/sources"

def test_m51_audit_report_exists():
    assert os.path.exists(os.path.join(REPORTS_DIR, "m5_source_audit.json"))

def test_m51_primary_sources_acquired():
    audit_file = os.path.join(REPORTS_DIR, "m5_source_audit.json")
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    status_map = {item["source_name"]: item.get("status") for item in data}
    
    assert status_map.get("Multitarget-CONAN") == "Acquired"
    assert status_map.get("CONAN") == "Acquired"

def test_m51_files_exist_with_size():
    # Multitarget-CONAN
    mt_conan_path = os.path.join(DATA_DIR, "Multitarget-CONAN.csv")
    assert os.path.exists(mt_conan_path)
    assert os.path.getsize(mt_conan_path) > 1000000 # > 1 MB
    
    # CONAN
    conan_path = os.path.join(DATA_DIR, "CONAN.json")
    assert os.path.exists(conan_path)
    assert os.path.getsize(conan_path) > 5000000 # > 5 MB
