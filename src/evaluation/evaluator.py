"""
CSHSM Milestone 10: Candidate Evaluator Interface & Deterministic Mock.

Defines:
- validate_scores: Strict validation function ensuring scores conform to 1-5 integer scale.
- BaseCandidateEvaluator: Shared abstract interface for candidate evaluation.
- MockCandidateEvaluator: Fully deterministic mock evaluator for testing without paid APIs.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from src.generation.fan_out import GeneratedCandidate, GenerationRequest
from src.evaluation.schema import CandidateEvaluation, EVALUATION_RUBRICS


def validate_scores(scores: Any) -> Tuple[bool, Optional[str], Dict[str, int]]:
    """
    Validates that candidate evaluation scores conform to the 1-5 integer scale.
    Returns (is_valid, error_reason, clean_scores_dict).
    """
    if not isinstance(scores, dict):
        return False, f"Scores must be a dictionary, got {type(scores).__name__}", {}

    required_criteria = {"relevance", "persuasiveness", "de_escalation", "safety"}
    if set(scores.keys()) != required_criteria:
        missing = required_criteria - set(scores.keys())
        extra = set(scores.keys()) - required_criteria
        return False, f"Scores keys mismatch: missing={missing}, extra={extra}", {}

    clean_scores: Dict[str, int] = {}
    for crit, val in scores.items():
        if isinstance(val, bool) or not isinstance(val, int):
            return False, f"Score for '{crit}' must be an integer, got {type(val).__name__} ({val})", {}
        if val < 1 or val > 5:
            return False, f"Score for '{crit}' must be between 1 and 5, got {val}", {}
        clean_scores[crit] = val

    return True, None, clean_scores


class BaseCandidateEvaluator(ABC):
    """
    Abstract interface for counter-speech candidate evaluators (LLM-as-a-judge).
    """

    @property
    @abstractmethod
    def evaluator_id(self) -> str:
        """Unique identifier for this evaluator instance."""
        pass

    @property
    @abstractmethod
    def evaluator_type(self) -> str:
        """Evaluator category ('MOCK' | 'LLM_JUDGE')."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Whether this evaluator is a mock/simulated test fixture."""
        pass

    @abstractmethod
    def evaluate_candidate(
        self,
        candidate: GeneratedCandidate,
        request: GenerationRequest
    ) -> CandidateEvaluation:
        """
        Evaluates a single counter-speech candidate across Relevance,
        Persuasiveness, De-escalation, and Safety.
        """
        pass

    def evaluate_batch(
        self,
        candidates: List[GeneratedCandidate],
        request: GenerationRequest
    ) -> List[CandidateEvaluation]:
        """
        Evaluates a list of candidates sequentially or concurrently.
        Catches exceptions per candidate to guarantee isolation.
        """
        results: List[CandidateEvaluation] = []
        for cand in candidates:
            try:
                eval_res = self.evaluate_candidate(cand, request)
                results.append(eval_res)
            except Exception as exc:
                now_iso = datetime.now(timezone.utc).isoformat()
                err_eval = CandidateEvaluation(
                    candidate_id=cand.candidate_id,
                    model_id=cand.model_id,
                    evaluator_id=self.evaluator_id,
                    evaluator_type=self.evaluator_type,
                    schema_version="1.0",
                    scores={},
                    reasons={"error": str(exc)},
                    is_mock=self.is_mock,
                    is_eligible=False,
                    ineligibility_reasons=[f"Evaluator exception: {type(exc).__name__}: {str(exc)}"],
                    aggregate_score=None,
                    evaluation_status="EVALUATION_ERROR",
                    timestamp=now_iso,
                    metadata={"error_type": type(exc).__name__}
                )
                results.append(err_eval)
        return results


class MockCandidateEvaluator(BaseCandidateEvaluator):
    """
    Deterministic mock evaluator for Milestone 10 testing and development.
    Explicitly marked as is_mock=True and evaluator_type='MOCK'.
    Does NOT call any external API or download models.
    """

    def __init__(
        self,
        evaluator_id: str = "mock-judge-v1",
        default_scores: Optional[Dict[str, int]] = None,
        candidate_overrides: Optional[Dict[str, Dict[str, int]]] = None,
        exception_to_raise: Optional[Exception] = None,
        force_malformed_scores: bool = False,
        force_status: Optional[str] = None
    ):
        self._evaluator_id = evaluator_id
        self.default_scores = default_scores or {
            "relevance": 4,
            "persuasiveness": 4,
            "de_escalation": 4,
            "safety": 5
        }
        self.candidate_overrides = candidate_overrides or {}
        self.exception_to_raise = exception_to_raise
        self.force_malformed_scores = force_malformed_scores
        self.force_status = force_status

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    @property
    def evaluator_type(self) -> str:
        return "MOCK"

    @property
    def is_mock(self) -> bool:
        return True

    def evaluate_candidate(
        self,
        candidate: GeneratedCandidate,
        request: GenerationRequest
    ) -> CandidateEvaluation:
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        if self.force_malformed_scores:
            # Emit malformed score dictionary
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                model_id=candidate.model_id,
                evaluator_id=self._evaluator_id,
                evaluator_type="MOCK",
                schema_version="1.0",
                scores={"relevance": 999, "safety": -5},  # Malformed out of range
                reasons={"error": "Forced malformed scores in mock judge"},
                is_mock=True,
                is_eligible=False,
                ineligibility_reasons=["Malformed evaluation scores"],
                aggregate_score=None,
                evaluation_status="MALFORMED_SCORES",
                timestamp=now_iso,
                metadata={"simulated": True}
            )

        # Check candidate overrides by candidate_id or model_id
        raw_scores = self.candidate_overrides.get(
            candidate.candidate_id,
            self.candidate_overrides.get(candidate.model_id, self.default_scores)
        )

        is_valid, err_msg, clean_scores = validate_scores(raw_scores)
        if not is_valid:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                model_id=candidate.model_id,
                evaluator_id=self._evaluator_id,
                evaluator_type="MOCK",
                schema_version="1.0",
                scores={},
                reasons={"validation_error": err_msg or "Invalid scores"},
                is_mock=True,
                is_eligible=False,
                ineligibility_reasons=[f"Invalid scores: {err_msg}"],
                aggregate_score=None,
                evaluation_status="MALFORMED_SCORES",
                timestamp=now_iso,
                metadata={"simulated": True}
            )

        # Generate deterministic mock reasons based on scores
        reasons = {
            "relevance": EVALUATION_RUBRICS["relevance"].anchors.get(clean_scores["relevance"], "Evaluated relevance."),
            "persuasiveness": EVALUATION_RUBRICS["persuasiveness"].anchors.get(clean_scores["persuasiveness"], "Evaluated persuasiveness."),
            "de_escalation": EVALUATION_RUBRICS["de_escalation"].anchors.get(clean_scores["de_escalation"], "Evaluated de-escalation."),
            "safety": EVALUATION_RUBRICS["safety"].anchors.get(clean_scores["safety"], "Evaluated safety.")
        }

        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            model_id=candidate.model_id,
            evaluator_id=self._evaluator_id,
            evaluator_type="MOCK",
            schema_version="1.0",
            scores=clean_scores,
            reasons=reasons,
            is_mock=True,
            is_eligible=True,  # Downstream ranker verifies mandatory hard gates
            ineligibility_reasons=[],
            aggregate_score=None,  # Computed by ranker according to configured weights
            evaluation_status=self.force_status or "SUCCESS",
            timestamp=now_iso,
            metadata={"simulated": True, "model_id": candidate.model_id}
        )
