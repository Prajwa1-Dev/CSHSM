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
    CandidateEvaluation,
    RankedCandidate,
    RankingConfig,
    RankingResult
)
from src.evaluation.evaluator import MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker
from src.safety.final_gate import (
    FinalGateResult,
    FinalSafetyGate,
    validate_final_candidate
)


# =====================================================================
# Fixtures & Helpers
# =====================================================================

def make_test_ranked_candidate(
    candidate_id: str = "GEN-test001",
    text: str = "We should respect diverse opinions and engage in peaceful dialogue.",
    model_id: str = "mock-gpt4o-mini",
    is_safe: bool = True,
    is_struct_valid: bool = True,
    safety_score: int = 5,
    relevance_score: int = 4,
    deesc_score: int = 4,
    pers_score: int = 4,
    eval_status: str = "SUCCESS",
    is_eligible: bool = True,
    is_mock: bool = True
) -> RankedCandidate:
    cand = GeneratedCandidate(
        candidate_id=candidate_id,
        text=text,
        model_id=model_id,
        provider="mock",
        is_mock=is_mock,
        query_text="Sample hate speech text",
        category="Religion",
        latency_ms=100.0,
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

    eval_res = CandidateEvaluation(
        candidate_id=candidate_id,
        model_id=model_id,
        evaluator_id="mock-judge-v1",
        evaluator_type="MOCK" if is_mock else "LLM_JUDGE",
        schema_version="1.0",
        scores={
            "relevance": relevance_score,
            "persuasiveness": pers_score,
            "de_escalation": deesc_score,
            "safety": safety_score
        },
        reasons={"relevance": "ok", "persuasiveness": "ok", "de_escalation": "ok", "safety": "ok"},
        is_mock=is_mock,
        is_eligible=is_eligible,
        ineligibility_reasons=[] if is_eligible else ["Gate failed"],
        aggregate_score=4.25,
        evaluation_status=eval_status,
        timestamp="2026-10-24T00:00:00Z"
    )

    return RankedCandidate(
        rank=1,
        candidate=cand,
        evaluation=eval_res,
        aggregate_score=4.25,
        normalized_score=0.8125
    )


def make_test_ranking_result(
    ranked_candidate: RankedCandidate,
    safe_fallback_triggered: bool = False
) -> RankingResult:
    return RankingResult(
        request=GenerationRequest(query_text="Sample query text", category="Religion"),
        evaluator_id="mock-judge-v1",
        evaluator_type="MOCK",
        is_mock=True,
        total_evaluated=1,
        eligible_count=1 if not safe_fallback_triggered else 0,
        ineligible_count=0 if not safe_fallback_triggered else 1,
        evaluations=[ranked_candidate.evaluation],
        ranked_candidates=[ranked_candidate] if not safe_fallback_triggered else [],
        ineligible_candidates=[] if not safe_fallback_triggered else [ranked_candidate.evaluation],
        selected_candidate=ranked_candidate if not safe_fallback_triggered else None,
        safe_fallback_triggered=safe_fallback_triggered,
        fallback_response="I cannot respond to this respectfully." if safe_fallback_triggered else None,
        status="SELECTED" if not safe_fallback_triggered else "ALL_INELIGIBLE",
        provenance={"timestamp": "2026-10-24T00:00:00Z"},
        disclaimer="EVALUATED CANDIDATE"
    )


# =====================================================================
# 1. Success & Mock Mode Policy Tests
# =====================================================================

def test_valid_candidate_passes_in_mock_mode():
    ranked_cand = make_test_ranked_candidate()
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "PASS_UNAPPROVED"
    assert result.safe_fallback_triggered is False
    assert result.text == ranked_cand.candidate.text.strip()
    assert result.candidate_id == "GEN-test001"
    assert result.is_mock_generation is True
    assert result.is_mock_evaluation is True
    assert result.fallback_response is None
    assert len(result.failure_reasons) == 0

    # Invariants strictly preserved
    assert result.quality_status == "generated_candidate"
    assert result.review_status == "unreviewed"
    assert result.human_audited is False
    assert "VALIDATED UNAPPROVED CANDIDATE" in result.disclaimer


def test_mock_evaluation_rejected_when_allow_mock_disabled():
    ranked_cand = make_test_ranked_candidate(is_mock=True)
    gate = FinalSafetyGate()

    # Production mode: allow_mock=False
    result = gate.validate(ranked_cand, allow_mock=False)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert result.text is None
    assert any("Mock or simulated" in r for r in result.failure_reasons)


# =====================================================================
# 2. Upstream Fallback & Missing Candidate Handling
# =====================================================================

def test_missing_selected_candidate_triggers_fallback():
    ranked_cand = make_test_ranked_candidate()
    ranking_res = make_test_ranking_result(ranked_cand, safe_fallback_triggered=True)
    gate = FinalSafetyGate()

    result = gate.validate(ranking_res, allow_mock=True)

    assert result.decision == "FALLBACK"
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert result.candidate_id is None
    assert result.text is None
    assert result.checks["upstream_selection"]["passed"] is False


# =====================================================================
# 3. Candidate Integrity & Metadata Consistency Tests
# =====================================================================

def test_empty_or_whitespace_text_fails_gate():
    ranked_cand = make_test_ranked_candidate(text="     \n  ")
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("empty or whitespace" in r for r in result.failure_reasons)


def test_candidate_id_mismatch_fails():
    ranked_cand = make_test_ranked_candidate()
    # Artificially corrupt evaluation candidate_id
    ranked_cand.evaluation.candidate_id = "GEN-DIFFERENT-ID"
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("Candidate ID mismatch" in r for r in result.failure_reasons)


def test_model_id_mismatch_fails():
    ranked_cand = make_test_ranked_candidate()
    ranked_cand.evaluation.model_id = "mock-claude-haiku"  # candidate has mock-gpt4o-mini
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("Model ID mismatch" in r for r in result.failure_reasons)


# =====================================================================
# 4. Upstream M9 Structural & Safety Validation Checks
# =====================================================================

def test_missing_or_failed_m9_structural_validation_fails():
    ranked_cand = make_test_ranked_candidate(is_struct_valid=False)
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("structural validation failed" in r for r in result.failure_reasons)


def test_failed_m9_safety_validation_fails():
    ranked_cand = make_test_ranked_candidate(is_safe=False)
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("safety validation failed" in r for r in result.failure_reasons)


# =====================================================================
# 5. M10 Evaluation Gates & Score Threshold Checks
# =====================================================================

def test_failed_m10_evaluation_status_fails():
    ranked_cand = make_test_ranked_candidate(eval_status="EVALUATION_ERROR")
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("evaluation status is not SUCCESS" in r for r in result.failure_reasons)


def test_failed_hard_safety_gate_fails():
    # Safety score = 2 (below minimum required threshold 4)
    ranked_cand = make_test_ranked_candidate(safety_score=2)
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("Safety score (2) below" in r for r in result.failure_reasons)


def test_failed_hard_relevance_gate_fails():
    # Relevance score = 2 (below minimum required threshold 3)
    ranked_cand = make_test_ranked_candidate(relevance_score=2)
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("Relevance score (2) below" in r for r in result.failure_reasons)


def test_failed_hard_de_escalation_gate_fails():
    # De-escalation score = 2 (below minimum required threshold 3)
    ranked_cand = make_test_ranked_candidate(deesc_score=2)
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("De-escalation score (2) below" in r for r in result.failure_reasons)


# =====================================================================
# 6. Independent Final Deterministic Safety Filter (Defense in Depth)
# =====================================================================

def test_final_deterministic_safety_filter_rejection():
    # Candidate text contains hate reinforcement or violent threats
    toxic_text = "I agree you are right that those people are terrible."
    ranked_cand = make_test_ranked_candidate(text=toxic_text)
    gate = FinalSafetyGate()

    result = gate.validate(ranked_cand, allow_mock=True)

    assert result.decision == "REJECTED"
    assert result.safe_fallback_triggered is True
    assert any("Independent final safety filter rejected text" in r for r in result.failure_reasons)


# =====================================================================
# 7. Serialization & Invariant Guarantees
# =====================================================================

def test_json_serialization():
    ranked_cand = make_test_ranked_candidate()
    gate = FinalSafetyGate()
    result = gate.validate(ranked_cand, allow_mock=True)

    d = result.to_dict()
    assert d["decision"] == "PASS_UNAPPROVED"
    assert d["candidate_id"] == "GEN-test001"
    assert d["quality_status"] == "generated_candidate"
    assert d["review_status"] == "unreviewed"
    assert d["human_audited"] is False

    json_str = json.dumps(d)
    assert "PASS_UNAPPROVED" in json_str
    assert "GEN-test001" in json_str


# =====================================================================
# 8. End-to-End M9 -> M10 -> M11 Pipeline Integration
# =====================================================================

def test_end_to_end_m9_m10_m11_integration():
    # Step 1: M9 Fan-Out
    fan_out = create_default_fan_out(use_mock=True)
    req = GenerationRequest(
        query_text="All immigrants from that nation are dangerous criminals.",
        category="Geo-political"
    )
    fan_out_res = fan_out.execute(req)
    assert fan_out_res.status == "SUCCESS"
    assert len(fan_out_res.safe_candidates) == 5

    # Step 2: M10 Evaluation & Ranking
    ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
    ranking_res = ranker.rank_fan_out(fan_out_res)
    assert ranking_res.status == "SELECTED"
    assert ranking_res.selected_candidate is not None

    # Step 3: M11 Final Safety Gate
    gate = FinalSafetyGate()
    final_res = gate.validate(ranking_res, allow_mock=True)

    assert final_res.decision == "PASS_UNAPPROVED"
    assert final_res.safe_fallback_triggered is False
    assert final_res.candidate_id is not None
    assert final_res.text is not None
    assert len(final_res.text) > 10
    assert final_res.quality_status == "generated_candidate"
    assert final_res.review_status == "unreviewed"
    assert final_res.human_audited is False


def test_all_ineligible_m10_flows_to_m11_fallback():
    # M10 evaluator rejects all candidates due to low safety scores
    unsafe_overrides = {
        f"mock-openai-gpt4o-mini": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-anthropic-claude3-haiku": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-google-gemini15-flash": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-mistral-small": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-cohere-command-r": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-gemma4-26b-it": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-glm-4.7-flash": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-gpt-oss-20b": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-qwen-3.8-27b": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2},
        f"mock-gpt-oss-120b": {"relevance": 5, "persuasiveness": 5, "de_escalation": 5, "safety": 2}
    }
    fan_out = create_default_fan_out(use_mock=True)
    req = GenerationRequest(query_text="Hate comment", category="Political")
    fan_out_res = fan_out.execute(req)

    ranker = CandidateRanker(evaluator=MockCandidateEvaluator(candidate_overrides=unsafe_overrides))
    ranking_res = ranker.rank_fan_out(fan_out_res)

    assert ranking_res.status == "ALL_INELIGIBLE"
    assert ranking_res.selected_candidate is None
    assert ranking_res.safe_fallback_triggered is True

    # Step 3: M11 Final Safety Gate receives fallback RankingResult
    final_res = validate_final_candidate(ranking_res, allow_mock=True)

    assert final_res.decision == "FALLBACK"
    assert final_res.safe_fallback_triggered is True
    assert final_res.fallback_response == "I cannot respond to this respectfully."
    assert final_res.text is None
