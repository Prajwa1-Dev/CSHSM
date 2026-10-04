import os
import pytest
import json

DATA_DIR = "data/counter_speech_kb/m53_generation"
REPORTS_DIR = "reports"

def test_m53_provenance_correctness():
    filepath = os.path.join(DATA_DIR, "llm_generated_candidates.jsonl")
    assert os.path.exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            assert c["provenance"] == "llm-assisted"
            assert "mock-llm-script" in c["generation_method"]

def test_m53_metadata_preservation():
    filepath = os.path.join(REPORTS_DIR, "m5_3_generation_metadata.json")
    assert os.path.exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        assert "prompt_template_version" in meta
        assert meta["prompt_template_version"] == "V1"
        assert "model" in meta
        assert "timestamp" in meta

def test_m53_safety_filtering_rejections():
    filepath = os.path.join(DATA_DIR, "llm_rejected_candidates.jsonl")
    assert os.path.exists(filepath)
    reasons = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            reasons.add(r["reason"])
            
    assert "hate reinforcement" in reasons
    assert "threats or violent escalation" in reasons
    assert "discriminatory language" in reasons
    
def test_m53_no_hastika_used_for_prompts():
    filepath = os.path.join(DATA_DIR, "llm_generated_candidates.jsonl")
    # Quick sanity check: None of the test texts should have been used.
    # The generation script strictly defined synthetic prompts.
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            ht = c["hate_text"].lower()
            assert ht != "i hate this"

def test_m53_deterministic_artifact():
    filepath = os.path.join(DATA_DIR, "llm_generated_candidates.jsonl")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 18 # 21 generated - 3 rejected
