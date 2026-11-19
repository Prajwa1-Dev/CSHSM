"""
CSHSM Milestone 10: Multi-Criteria Candidate Ranker & Selection Engine.

Flow:
  Safe Candidates from M9 FanOutResult
      ↓
  Independent Structural & Safety Invariant Verification
      ↓
  Multi-Criteria Evaluation (Relevance, Persuasiveness, De-escalation, Safety)
      ↓
  Mandatory Hard Gates Enforcement (Safety >= min_safety, Relevance >= min_rel, etc.)
      ↓
  Weighted Scoring & Deterministic Tie-Breaking
      ↓
  Top-1 Candidate Selection or Safe Fallback Trigger
"""

import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple

from src.generation.fan_out import GeneratedCandidate, GenerationRequest, FanOutResult
from src.evaluation.schema import (
    CandidateEvaluation,
    RankedCandidate,
    RankingConfig,
    RankingResult
)
from src.evaluation.evaluator import BaseCandidateEvaluator, validate_scores


class CandidateRanker:
    """
    Milestone 10 Candidate Ranker and Selection Engine.

    Evaluates candidates from M9, enforces mandatory hard safety and quality gates,
    computes weighted aggregate scores, performs deterministic tie-breaking,
    and selects the best candidate (or triggers fallback if no candidate qualifies).
    """

    DISCLAIMER_EVALUATED = (
        "EVALUATED CANDIDATE — Evaluated and ranked by M10 evaluation layer. "
        "Unapproved pending downstream safety gating and formal review policy."
    )

    def __init__(
        self,
        evaluator: BaseCandidateEvaluator,
        config: Optional[RankingConfig] = None
    ):
        if evaluator is None:
            raise ValueError("evaluator cannot be None")
        self.evaluator = evaluator
        self.config = config or RankingConfig()
        self.config.validate()

    def rank_fan_out(
        self,
        fan_out_result: FanOutResult,
        query_override: Optional[str] = None
    ) -> RankingResult:
        """
        Convenience method ingesting an M9 FanOutResult directly.
        """
        candidates = fan_out_result.safe_candidates
        return self.rank_candidates(
            candidates=candidates,
            request=fan_out_result.request
        )

    def rank_candidates(
        self,
        candidates: List[GeneratedCandidate],
        request: GenerationRequest
    ) -> RankingResult:
        """
        Ranks candidate counter-speech responses and selects the highest-scoring
        candidate that passes all mandatory safety and evaluation gates.
        """
        start_time = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Handle empty candidate pool
        if not candidates:
            return RankingResult(
                request=request,
                evaluator_id=self.evaluator.evaluator_id,
                evaluator_type=self.evaluator.evaluator_type,
                is_mock=self.evaluator.is_mock,
                total_evaluated=0,
                eligible_count=0,
                ineligible_count=0,
                evaluations=[],
                ranked_candidates=[],
                ineligible_candidates=[],
                selected_candidate=None,
                safe_fallback_triggered=True,
                fallback_response=self.config.fallback_message,
                status="EMPTY_POOL",
                provenance={
                    "evaluator_id": self.evaluator.evaluator_id,
                    "is_mock": self.evaluator.is_mock,
                    "timestamp": now_iso,
                    "reason": "Candidate pool was empty."
                },
                disclaimer="SAFE FALLBACK — No safe candidates available for evaluation."
            )

        # 1. Independent Upstream Verification
        # Ensure candidate passed M9 structural validation and safety filter
        candidates_to_evaluate: List[GeneratedCandidate] = []
        upstream_ineligible_evals: List[CandidateEvaluation] = []
        candidate_map: Dict[str, GeneratedCandidate] = {}

        for cand in candidates:
            candidate_map[cand.candidate_id] = cand

            is_struct_valid = cand.structural_validation.get("is_valid", False)
            is_safe_valid = cand.safety_validation.get("is_safe", False)

            if not is_struct_valid or not is_safe_valid:
                reasons = []
                if not is_struct_valid:
                    reasons.append(f"Failed structural validation: {cand.structural_validation.get('validation_reason')}")
                if not is_safe_valid:
                    reasons.append(f"Failed upstream safety filter: {cand.safety_validation.get('rejection_reason')}")

                upstream_eval = CandidateEvaluation(
                    candidate_id=cand.candidate_id,
                    model_id=cand.model_id,
                    evaluator_id=self.evaluator.evaluator_id,
                    evaluator_type=self.evaluator.evaluator_type,
                    schema_version="1.0",
                    scores={},
                    reasons={},
                    is_mock=self.evaluator.is_mock,
                    is_eligible=False,
                    ineligibility_reasons=reasons,
                    aggregate_score=None,
                    evaluation_status="UPSTREAM_INELIGIBLE",
                    timestamp=now_iso,
                    metadata={"upstream_rejection": True}
                )
                upstream_ineligible_evals.append(upstream_eval)
            else:
                candidates_to_evaluate.append(cand)

        # 2. Run Evaluator on Upstream-Valid Candidates
        evaluated_results = self.evaluator.evaluate_batch(candidates_to_evaluate, request)
        all_evaluations: List[CandidateEvaluation] = upstream_ineligible_evals + evaluated_results

        # 3. Enforce Mandatory Hard Gates & Compute Aggregate Scores
        ranked_candidates: List[RankedCandidate] = []
        ineligible_candidates: List[CandidateEvaluation] = list(upstream_ineligible_evals)

        for eval_res in evaluated_results:
            cand = candidate_map[eval_res.candidate_id]
            is_eligible, inelig_reasons = self._check_mandatory_gates(eval_res)

            eval_res.is_eligible = is_eligible
            eval_res.ineligibility_reasons = inelig_reasons

            if not is_eligible:
                eval_res.aggregate_score = None
                ineligible_candidates.append(eval_res)
                continue

            # Compute weighted aggregate score
            agg_score = self._compute_aggregate_score(eval_res.scores)
            eval_res.aggregate_score = round(agg_score, 4)
            norm_score = round((agg_score - 1.0) / 4.0, 4)

            # Preserve unapproved invariants
            cand.disclaimer = self.DISCLAIMER_EVALUATED

            ranked_item = RankedCandidate(
                rank=0,  # Assigned after sorting
                candidate=cand,
                evaluation=eval_res,
                aggregate_score=agg_score,
                normalized_score=norm_score
            )
            ranked_candidates.append(ranked_item)

        # 4. Deterministic Tie-Breaking & Rank Ordering
        # Tie-breaker hierarchy:
        #   1. Aggregate score (descending)
        #   2. Safety score (descending)
        #   3. De-escalation score (descending)
        #   4. Relevance score (descending)
        #   5. Candidate ID (ascending - deterministic tie-breaker)
        ranked_candidates.sort(
            key=lambda item: (
                -round(item.aggregate_score, 4),
                -item.evaluation.scores.get("safety", 0),
                -item.evaluation.scores.get("de_escalation", 0),
                -item.evaluation.scores.get("relevance", 0),
                item.candidate.candidate_id
            )
        )

        for idx, item in enumerate(ranked_candidates):
            item.rank = idx + 1

        # 5. Selection & Safe Fallback Trigger
        selected_candidate = ranked_candidates[0] if ranked_candidates else None
        safe_fallback_triggered = (selected_candidate is None)
        fallback_response = self.config.fallback_message if safe_fallback_triggered else None

        if selected_candidate:
            status = "SELECTED"
        elif not candidates_to_evaluate:
            status = "ALL_INELIGIBLE"
        elif all(e.evaluation_status == "EVALUATION_ERROR" for e in evaluated_results):
            status = "EVALUATION_FAILED"
        else:
            status = "ALL_INELIGIBLE"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        provenance = {
            "evaluator_id": self.evaluator.evaluator_id,
            "evaluator_type": self.evaluator.evaluator_type,
            "is_mock": self.evaluator.is_mock,
            "latency_ms": elapsed_ms,
            "timestamp": now_iso,
            "weights": self.config.weights,
            "thresholds": {
                "min_safety": self.config.min_safety_score,
                "min_relevance": self.config.min_relevance_score,
                "min_de_escalation": self.config.min_de_escalation_score,
                "min_persuasiveness": self.config.min_persuasiveness_score
            }
        }

        return RankingResult(
            request=request,
            evaluator_id=self.evaluator.evaluator_id,
            evaluator_type=self.evaluator.evaluator_type,
            is_mock=self.evaluator.is_mock,
            total_evaluated=len(all_evaluations),
            eligible_count=len(ranked_candidates),
            ineligible_count=len(ineligible_candidates),
            evaluations=all_evaluations,
            ranked_candidates=ranked_candidates,
            ineligible_candidates=ineligible_candidates,
            selected_candidate=selected_candidate,
            safe_fallback_triggered=safe_fallback_triggered,
            fallback_response=fallback_response,
            status=status,
            provenance=provenance,
            disclaimer=self.DISCLAIMER_EVALUATED
        )

    def _check_mandatory_gates(
        self,
        eval_res: CandidateEvaluation
    ) -> Tuple[bool, List[str]]:
        """
        Enforces mandatory hard gates:
          - Evaluator status must be SUCCESS
          - Scores must be valid integers in [1, 5]
          - Safety score >= min_safety_score
          - Relevance score >= min_relevance_score
          - De-escalation score >= min_de_escalation_score
          - Persuasiveness score >= min_persuasiveness_score

        A high aggregate score CANNOT override a failed gate.
        """
        reasons: List[str] = list(eval_res.ineligibility_reasons)

        if eval_res.evaluation_status != "SUCCESS":
            reasons.append(f"Evaluation failed with status: {eval_res.evaluation_status}")
            return False, reasons

        is_valid, err_msg, clean_scores = validate_scores(eval_res.scores)
        if not is_valid:
            reasons.append(f"Score validation failure: {err_msg}")
            return False, reasons

        safety_val = clean_scores.get("safety", 0)
        rel_val = clean_scores.get("relevance", 0)
        deesc_val = clean_scores.get("de_escalation", 0)
        pers_val = clean_scores.get("persuasiveness", 0)

        # Mandatory Hard Safety Gate
        if safety_val < self.config.min_safety_score:
            reasons.append(
                f"Safety score ({safety_val}) failed mandatory threshold ({self.config.min_safety_score})."
            )

        # Mandatory Hard Relevance Gate
        if rel_val < self.config.min_relevance_score:
            reasons.append(
                f"Relevance score ({rel_val}) failed mandatory threshold ({self.config.min_relevance_score})."
            )

        # Mandatory Hard De-escalation Gate
        if deesc_val < self.config.min_de_escalation_score:
            reasons.append(
                f"De-escalation score ({deesc_val}) failed mandatory threshold ({self.config.min_de_escalation_score})."
            )

        # Mandatory Hard Persuasiveness Gate
        if pers_val < self.config.min_persuasiveness_score:
            reasons.append(
                f"Persuasiveness score ({pers_val}) failed mandatory threshold ({self.config.min_persuasiveness_score})."
            )

        is_eligible = (len(reasons) == 0)
        return is_eligible, reasons

    def _compute_aggregate_score(self, scores: Dict[str, int]) -> float:
        """Computes weighted aggregate score."""
        return sum(
            self.config.weights[crit] * scores[crit]
            for crit in self.config.weights
        )
