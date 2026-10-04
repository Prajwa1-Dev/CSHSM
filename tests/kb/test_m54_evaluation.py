import os
import sys
import pytest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.kb.schema import validate_entry, KBEntry

DATA_DIR = "data/counter_speech_kb"

def test_m54_human_review_required_state():
    path = os.path.join(DATA_DIR, "human_review_required.jsonl")
    assert os.path.exists(path)
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            assert c["provenance"] == "llm-assisted"
            assert c["quality_status"] == "human_review_required"
            count += 1
    assert count == 18 # All 18 M5.3 candidates must be here

def test_m54_eligible_deferred_state_exists():
    path = os.path.join(DATA_DIR, "eligible_deferred.jsonl")
    assert os.path.exists(path)

def test_m54_provenance_preservation():
    path = os.path.join(DATA_DIR, "human_review_required.jsonl")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                assert c["provenance"] == "llm-assisted"
                assert "mock-llm-script" in c["generation_method"]

def test_m54_final_kb_invariants():
    path = os.path.join(DATA_DIR, "approved_kb_v2.jsonl")
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 0 # Blocked by lack of real LLM

def test_m54_blocked_eval_reporting():
    path = os.path.join("reports", "m5_4_evaluation_v2.json")
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        rep = json.load(f)
        assert "blocker" in rep
        assert rep["evaluations_real"] == 0
        assert rep["evaluations_simulated_in_v2"] == 0
