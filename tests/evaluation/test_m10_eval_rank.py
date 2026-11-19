import urllib.request
import os
import sys
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.fan_out import (
    GenerationRequest,
    GeneratedCandidate,
    FanOutResult,
    create_default_fan_out
)
from src.evaluation.schema import (
    CriterionDefinition,
    EVALUATION_RUBRICS,
    CandidateEvaluation,
    RankedCandidate,
    RankingConfig,
    RankingResult
)
from src.evaluation.evaluator import (
    BaseCandidateEvaluator,
    MockCandidateEvaluator,
    GeminiCandidateEvaluator,
    validate_scores
)
from src.evaluation.ranker import CandidateRanker


# =====================================================================
# Fixtures & Helpers
# =====================================================================

def make_test_candidate(
    candidate_id: str = "GEN-test001",
    text: str = "We can debate policy without resorting to insults.",
    model_id: str = "mock-gpt4o-mini",
    is_safe: bool = True,
    is_struct_valid: bool = True,
    provider: str = "mock"
) -> GeneratedCandidate:
    return GeneratedCandidate(
        candidate_id=candidate_id,
        text=text,
        model_id=model_id,
        provider=provider,
        is_mock=True,
        query_text="Sample hate context",
        category="Political",
        latency_ms=120.0,
        quality_status="generated_candidate",
        review_status="unreviewed",
        human_audited=False,
        structural_validation={
            "is_valid": is_struct_valid,
            "length_chars": len(text),
            "length_tokens_approx": len(text.split()),
            "is_empty": False,
            "is_excessively_long": False,
            "is_duplicate": False,
            "validation_reason": None if is_struct_valid else "INVALID"
        },
        safety_validation={
            "is_safe": is_safe,
            "rejection_reason": "" if is_safe else "hate reinforcement"
        },
        created_at="2026-10-24T00:00:00Z",
        provenance={"slot": 0, "model_id": model_id},
        disclaimer="UNREVIEWED GENERATED CANDIDATE"
    )


# =====================================================================
# 1. Criteria & Score Validation Tests
# =====================================================================

def test_validate_scores_valid():
    valid = {"relevance": 5, "persuasiveness": 4, "de_escalation": 4, "safety": 5}
    is_valid, err, clean = validate_scores(valid)
    assert is_valid is True
    assert err is None
    assert clean == valid


def test_validate_scores_missing_key():
    missing = {"relevance": 4, "persuasiveness": 3, "de_escalation": 4}  # missing safety
    is_valid, err, _ = validate_scores(missing)
    assert is_valid is False
    assert "missing=" in err


def test_validate_scores_extra_key():
    extra = {"relevance": 4, "persuasiveness": 3, "de_escalation": 4, "safety": 5, "toxicity": 1}
    is_valid, err, _ = validate_scores(extra)
    assert is_valid is False
    assert "extra=" in err


def test_validate_scores_out_of_range():
    too_high = {"relevance": 6, "persuasiveness": 3, "de_escalation": 4, "safety": 5}
    is_valid, err, _ = validate_scores(too_high)
    assert is_valid is False
    assert "between 1 and 5" in err

    too_low = {"relevance": 0, "persuasiveness": 3, "de_escalation": 4, "safety": 5}
    is_valid, err, _ = validate_scores(too_low)
    assert is_valid is False
    assert "between 1 and 5" in err


def test_validate_scores_invalid_types():
    # Float rejected (must be integer scale)
    float_val = {"relevance": 4.5, "persuasiveness": 3, "de_escalation": 4, "safety": 5}
    is_valid, err, _ = validate_scores(float_val)
    assert is_valid is False
    assert "must be an integer" in err

    # Boolean rejected
    bool_val = {"relevance": True, "persuasiveness": 3, "de_escalation": 4, "safety": 5}
    is_valid, err, _ = validate_scores(bool_val)
    assert is_valid is False
    assert "must be an integer" in err


def test_rubrics_completeness():
    assert len(EVALUATION_RUBRICS) == 4
    for crit in ["relevance", "persuasiveness", "de_escalation", "safety"]:
        assert crit in EVALUATION_RUBRICS
        rubric = EVALUATION_RUBRICS[crit]
        assert rubric.min_score == 1
        assert rubric.max_score == 5
        assert len(rubric.anchors) == 5


# =====================================================================
# 2. Safety Hard Gate Overrides High Aggregate Score
# =====================================================================

def test_safety_failure_overrides_high_aggregate_score():
    # Candidate with stellar relevance, persuasiveness, de-escalation, but fails safety (safety = 2 < 4)
    c1 = make_test_candidate(candidate_id="cand-stellar-unsafe", text="Very persuasive but unsafe argument.")
    c2 = make_test_candidate(candidate_id="cand-moderate-safe", text="Moderate counter speech that is safe.")

    overrides = {
        "cand-stellar-unsafe": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        "cand-moderate-safe": {"relevance": 4, "persuasiveness": 3, "de_escalation": 3, "safety": 5}
    }
    evaluator = MockCandidateEvaluator(candidate_overrides=overrides)
    ranker = CandidateRanker(evaluator=evaluator)

    req = GenerationRequest(query_text="Hate context", category="Political")
    result = ranker.rank_candidates([c1, c2], req)

    # c1 has aggregate score 5*0.35 + 5*0.25 + 5*0.25 + 2*0.15 = 4.55
    # c2 has aggregate score 4*0.35 + 3*0.25 + 3*0.25 + 5*0.15 = 3.65
    # BUT c1 MUST be disqualified by the hard safety gate!
    assert result.status == "SELECTED"
    assert result.selected_candidate is not None
    assert result.selected_candidate.candidate.candidate_id == "cand-moderate-safe"

    # Confirm c1 was recorded in ineligible candidates
    inelig_ids = [e.candidate_id for e in result.ineligible_candidates]
    assert "cand-stellar-unsafe" in inelig_ids
    c1_eval = [e for e in result.ineligible_candidates if e.candidate_id == "cand-stellar-unsafe"][0]
    assert c1_eval.is_eligible is False
    assert any("Safety score" in r for r in c1_eval.ineligibility_reasons)


# =====================================================================
# 3. Evaluator Exceptions & Isolation
# =====================================================================

def test_evaluator_exception_isolated():
    c1 = make_test_candidate(candidate_id="cand-good", text="Good response.")
    c2 = make_test_candidate(candidate_id="cand-crash", text="Will crash judge.")

    class FaultyEvaluator(BaseCandidateEvaluator):
        @property
        def evaluator_id(self) -> str: return "faulty-judge"
        @property
        def evaluator_type(self) -> str: return "MOCK"
        @property
        def is_mock(self) -> bool: return True

        def evaluate_candidate(self, cand: GeneratedCandidate, req: GenerationRequest) -> CandidateEvaluation:
            if cand.candidate_id == "cand-crash":
                raise RuntimeError("LLM API parsing fatal error")
            return CandidateEvaluation(
                candidate_id=cand.candidate_id,
                model_id=cand.model_id,
                evaluator_id="faulty-judge",
                evaluator_type="MOCK",
                schema_version="1.0",
                scores={"relevance": 4, "persuasiveness": 4, "de_escalation": 4, "safety": 5},
                reasons={"relevance": "ok", "persuasiveness": "ok", "de_escalation": "ok", "safety": "ok"},
                is_mock=True,
                is_eligible=True,
                ineligibility_reasons=[],
                aggregate_score=None,
                evaluation_status="SUCCESS",
                timestamp="2026-10-24T00:00:00Z"
            )

    ranker = CandidateRanker(evaluator=FaultyEvaluator())
    req = GenerationRequest(query_text="Hate context")
    result = ranker.rank_candidates([c1, c2], req)

    # c2 crash did NOT crash the ranker or prevent c1 from being selected!
    assert result.status == "SELECTED"
    assert result.selected_candidate.candidate.candidate_id == "cand-good"

    # c2 recorded as EVALUATION_ERROR
    c2_eval = [e for e in result.evaluations if e.candidate_id == "cand-crash"][0]
    assert c2_eval.evaluation_status == "EVALUATION_ERROR"
    assert c2_eval.is_eligible is False
    assert "LLM API parsing fatal error" in c2_eval.ineligibility_reasons[0]


# =====================================================================
# 4. Empty Pool & All-Ineligible Fallback
# =====================================================================

def test_empty_candidate_pool_triggers_fallback():
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    req = GenerationRequest(query_text="Empty pool test")
    result = ranker.rank_candidates([], req)

    assert result.status == "EMPTY_POOL"
    assert result.selected_candidate is None
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert len(result.ranked_candidates) == 0


def test_all_ineligible_candidates_trigger_fallback():
    # Both candidates fail hard safety gate
    c1 = make_test_candidate(candidate_id="c1", text="Unsafe text 1")
    c2 = make_test_candidate(candidate_id="c2", text="Unsafe text 2")

    overrides = {
        "c1": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        "c2": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 3}
    }
    evaluator = MockCandidateEvaluator(candidate_overrides=overrides)
    ranker = CandidateRanker(evaluator=evaluator)

    result = ranker.rank_candidates([c1, c2], GenerationRequest(query_text="Test"))

    assert result.status == "ALL_INELIGIBLE"
    assert result.selected_candidate is None
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert len(result.ineligible_candidates) == 2


def test_upstream_ineligible_candidates_rejected_without_judge():
    # Candidate failed M9 safety filter
    unsafe_cand = make_test_candidate(candidate_id="c-unsafe", text="Toxic text", is_safe=False)
    # Candidate failed M9 structural validation
    invalid_cand = make_test_candidate(candidate_id="c-invalid", text="Bad", is_struct_valid=False)

    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    result = ranker.rank_candidates([unsafe_cand, invalid_cand], GenerationRequest(query_text="Test"))

    assert result.status == "ALL_INELIGIBLE"
    assert result.selected_candidate is None
    for e in result.ineligible_candidates:
        assert e.evaluation_status == "UPSTREAM_INELIGIBLE"
        assert e.is_eligible is False


# =====================================================================
# 5. Deterministic Ranking & Tie-Breaking
# =====================================================================

def test_deterministic_ranking_by_score():
    c_high = make_test_candidate(candidate_id="c-high", text="High quality response")
    c_mid = make_test_candidate(candidate_id="c-mid", text="Medium quality response")
    c_low = make_test_candidate(candidate_id="c-low", text="Low quality response")

    overrides = {
        "c-high": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 5},
        "c-mid":  {"relevance": 4, "persuasiveness": 4, "de_escalation": 4, "safety": 4},
        "c-low":  {"relevance": 3, "persuasiveness": 3, "de_escalation": 3, "safety": 4}
    }
    evaluator = MockCandidateEvaluator(candidate_overrides=overrides)
    ranker = CandidateRanker(evaluator=evaluator)

    result = ranker.rank_candidates([c_low, c_high, c_mid], GenerationRequest(query_text="Test"))

    assert result.status == "SELECTED"
    assert len(result.ranked_candidates) == 3

    assert result.ranked_candidates[0].candidate.candidate_id == "c-high"
    assert result.ranked_candidates[0].rank == 1

    assert result.ranked_candidates[1].candidate.candidate_id == "c-mid"
    assert result.ranked_candidates[1].rank == 2

    assert result.ranked_candidates[2].candidate.candidate_id == "c-low"
    assert result.ranked_candidates[2].rank == 3


def test_deterministic_tie_breaking():
    # Two candidates with IDENTICAL aggregate score:
    # c_safe_first: relevance=3, persuasiveness=4, de_escalation=4, safety=5
    # c_rel_first:  relevance=5, persuasiveness=4, de_escalation=4, safety=4 (wait, check weights)
    # Let's make weights equal for relevance and safety: 0.25 each
    equal_config = RankingConfig(
        weights={"relevance": 0.25, "persuasiveness": 0.25, "de_escalation": 0.25, "safety": 0.25}
    )

    # c_a: rel=4, pers=4, deesc=4, safe=5 -> sum = 17 / 4 = 4.25
    # c_b: rel=5, pers=4, deesc=4, safe=4 -> sum = 17 / 4 = 4.25
    # Both have aggregate score = 4.25!
    # Tie-breaker hierarchy: Higher safety score first! c_a has safety=5 > c_b safety=4.
    c_a = make_test_candidate(candidate_id="c-cand-A", text="Response A")
    c_b = make_test_candidate(candidate_id="c-cand-B", text="Response B")

    overrides = {
        "c-cand-A": {"relevance": 4, "persuasiveness": 4, "de_escalation": 4, "safety": 5},
        "c-cand-B": {"relevance": 5, "persuasiveness": 4, "de_escalation": 4, "safety": 4}
    }
    evaluator = MockCandidateEvaluator(candidate_overrides=overrides)
    ranker = CandidateRanker(evaluator=evaluator, config=equal_config)

    result = ranker.rank_candidates([c_b, c_a], GenerationRequest(query_text="Test"))

    assert result.ranked_candidates[0].candidate.candidate_id == "c-cand-A"
    assert result.ranked_candidates[1].candidate.candidate_id == "c-cand-B"


def test_tie_breaking_alphabetical_fallback():
    # Two candidates with identical scores across all 4 criteria
    c_zebra = make_test_candidate(candidate_id="GEN-zebra", text="Zebra text")
    c_alpha = make_test_candidate(candidate_id="GEN-alpha", text="Alpha text")

    evaluator = MockCandidateEvaluator()  # Default identical scores
    ranker = CandidateRanker(evaluator=evaluator)

    result = ranker.rank_candidates([c_zebra, c_alpha], GenerationRequest(query_text="Test"))

    # Tie-breaker 5: candidate_id ascending -> GEN-alpha comes first
    assert result.ranked_candidates[0].candidate.candidate_id == "GEN-alpha"
    assert result.ranked_candidates[1].candidate.candidate_id == "GEN-zebra"


# =====================================================================
# 6. Configuration Validation
# =====================================================================

def test_ranking_config_validations():
    # Weights sum != 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        RankingConfig(weights={"relevance": 0.5, "persuasiveness": 0.5, "de_escalation": 0.5, "safety": 0.5}).validate()

    # Negative weight
    with pytest.raises(ValueError, match="non-negative float"):
        RankingConfig(weights={"relevance": 1.2, "persuasiveness": -0.2, "de_escalation": 0.0, "safety": 0.0}).validate()

    # Missing criterion in weights
    with pytest.raises(ValueError, match="Weights keys must match"):
        RankingConfig(weights={"relevance": 0.5, "persuasiveness": 0.5}).validate()

    # Out of range threshold (<1 or >5)
    with pytest.raises(ValueError, match="min_safety_score must be an integer between 1 and 5"):
        RankingConfig(min_safety_score=6).validate()


# =====================================================================
# 7. Unapproved Candidate Invariants & Disclaimer
# =====================================================================

def test_selected_candidate_preserves_unapproved_status():
    cand = make_test_candidate(candidate_id="test-unapproved")
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    result = ranker.rank_candidates([cand], GenerationRequest(query_text="Test"))

    assert result.status == "SELECTED"
    selected = result.selected_candidate.candidate

    # Invariants strictly preserved
    assert selected.quality_status == "generated_candidate"
    assert selected.review_status == "unreviewed"
    assert selected.human_audited is False
    assert "Unapproved" in selected.disclaimer


# =====================================================================
# 8. Ingesting M9 FanOutResult End-to-End
# =====================================================================

def test_rank_fan_out_end_to_end():
    fan_out = create_default_fan_out(use_mock=True)
    req = GenerationRequest(query_text="People from that region are completely untrustworthy.", category="Geo-political")
    fan_out_result = fan_out.execute(req)

    assert fan_out_result.status == "SUCCESS"
    assert len(fan_out_result.safe_candidates) == 5

    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    ranking_result = ranker.rank_fan_out(fan_out_result)

    assert ranking_result.status == "SELECTED"
    assert ranking_result.selected_candidate is not None
    assert ranking_result.total_evaluated == 5
    assert ranking_result.eligible_count == 5
    assert len(ranking_result.ranked_candidates) == 5
    assert ranking_result.is_mock is True
    assert ranking_result.evaluator_type == "MOCK"


# =====================================================================
# 9. JSON Serialization Tests
# =====================================================================

def test_ranking_result_json_serialization():
    cand = make_test_candidate(candidate_id="c-serial")
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    result = ranker.rank_candidates([cand], GenerationRequest(query_text="Serialization test"))

    d = result.to_dict()
    assert d["status"] == "SELECTED"
    assert d["selected_candidate"]["candidate"]["candidate_id"] == "c-serial"
    assert d["selected_candidate"]["rank"] == 1

    json_str = json.dumps(d)
    assert "c-serial" in json_str
    assert "MOCK" in json_str


# =====================================================================
# 10. Gemini Flash-Lite LLM Judge & Self-Evaluation Guard Tests
# =====================================================================

def test_gemini_evaluator_unconfigured_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    evaluator = GeminiCandidateEvaluator(model_id="gemini-flash-lite-latest")
    cand = make_test_candidate(candidate_id="c-groq", provider="groq", model_id="openai/gpt-oss-20b")
    req = GenerationRequest(query_text="Sample hate speech text")

    res = evaluator.evaluate_candidate(cand, req)
    assert res.is_eligible is False
    assert res.evaluation_status == "EVALUATION_ERROR"
    assert "GOOGLE_API_KEY" in res.ineligibility_reasons[0]
    assert res.evaluator_type == "LLM_JUDGE"
    assert res.is_mock is False


def test_gemini_evaluator_self_evaluation_rejection():
    evaluator = GeminiCandidateEvaluator(model_id="gemini-flash-lite-latest")
    req = GenerationRequest(query_text="Sample hate speech text")

    # Rejection 1: candidate provider is "google"
    cand_google = make_test_candidate(candidate_id="c-self-1", provider="google", model_id="gemini-flash-lite")
    with pytest.raises(ValueError, match="Self-evaluation violation"):
        evaluator.evaluate_candidate(cand_google, req)

    # Rejection 2: candidate model_id contains "gemini"
    cand_gemini = make_test_candidate(candidate_id="c-self-2", provider="mock", model_id="mock-gemini-model")
    with pytest.raises(ValueError, match="Self-evaluation violation"):
        evaluator.evaluate_candidate(cand_gemini, req)


def test_gemini_evaluator_malformed_json_fails_closed(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-gemini-key")
    evaluator = GeminiCandidateEvaluator(model_id="gemini-flash-lite-latest")
    cand = make_test_candidate(candidate_id="c-groq-1", provider="groq", model_id="qwen/qwen3.8-27b")
    req = GenerationRequest(query_text="Sample hate text")

    # Mock response containing invalid scores (out of 1-5 range)
    mock_payload = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "scores": {"relevance": 99, "persuasiveness": 3, "de_escalation": 3, "safety": 5},
                        "reasons": {"relevance": "Out of range"}
                    })
                }]
            }
        }]
    }

    class MockHTTPResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(mock_payload).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse())

    res = evaluator.evaluate_candidate(cand, req)
    assert res.is_eligible is False
    assert res.evaluation_status == "MALFORMED_SCORES"
    assert any("Invalid scores" in r for r in res.ineligibility_reasons)


def test_gemini_evaluator_valid_scores_passes(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-gemini-key")
    evaluator = GeminiCandidateEvaluator(model_id="gemini-flash-lite-latest")
    cand = make_test_candidate(candidate_id="c-groq-2", provider="groq", model_id="openai/gpt-oss-120b")
    req = GenerationRequest(query_text="Sample hate text")

    mock_payload = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "scores": {
                            "relevance": 4,
                            "persuasiveness": 4,
                            "de_escalation": 5,
                            "safety": 5
                        },
                        "reasons": {
                            "relevance": "Directly counters hateful claim.",
                            "persuasiveness": "Logical and constructive.",
                            "de_escalation": "Calm and empathetic tone.",
                            "safety": "Completely avoids toxic phrases."
                        }
                    })
                }]
            }
        }]
    }

    class MockHTTPResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(mock_payload).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse())

    res = evaluator.evaluate_candidate(cand, req)
    assert res.is_eligible is True
    assert res.evaluation_status == "SUCCESS"
    assert res.scores["relevance"] == 4
    assert res.scores["persuasiveness"] == 4
    assert res.scores["de_escalation"] == 5
    assert res.scores["safety"] == 5
    assert res.evaluator_type == "LLM_JUDGE"
    assert res.is_mock is False
    assert "Directly counters" in res.reasons["relevance"]
