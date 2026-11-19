"""
CSHSM Milestone 10: Evaluation & Ranking Schema.

Defines:
- CriterionDefinition: Structured definitions, anchors, and rubrics (1-5 scale)
  for Relevance, Persuasiveness, De-escalation, and Safety.
- CandidateEvaluation: Per-candidate evaluation result capturing scores, reasons,
  eligibility gates, and simulated flags.
- RankedCandidate: Container binding a GeneratedCandidate to its CandidateEvaluation,
  with explicit rank position, aggregate score, and normalized score.
- RankingConfig: Configuration for criteria weights, hard safety/quality thresholds,
  and fallback message.
- RankingResult: Complete outcome container for an evaluation and ranking run.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from src.generation.fan_out import GeneratedCandidate, GenerationRequest


# =====================================================================
# Criterion Rubric Definitions
# =====================================================================

@dataclass(frozen=True)
class CriterionDefinition:
    """Metadata and 1-5 scoring rubric anchors for an evaluation criterion."""
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5
    anchors: Dict[int, str] = field(default_factory=dict)


EVALUATION_RUBRICS: Dict[str, CriterionDefinition] = {
    "relevance": CriterionDefinition(
        name="relevance",
        description="Whether the counter-speech directly addresses the specific hateful claim or context.",
        anchors={
            1: "Completely irrelevant, generic, or off-topic.",
            2: "Vague or tangential connection to the hateful context.",
            3: "Moderately addresses the premise with some generic phrasing.",
            4: "Directly addresses the hateful claim with clear context awareness.",
            5: "Precisely refutes or addresses the specific hateful claim with tailored context awareness."
        }
    ),
    "persuasiveness": CriterionDefinition(
        name="persuasiveness",
        description="Whether the counter-speech is logical, constructive, understandable, and persuasive.",
        anchors={
            1: "Unconvincing, incoherent, or purely reactive.",
            2: "Weak argument with little constructive value.",
            3: "Coherent and logical, presenting a reasonable perspective.",
            4: "Compelling, constructive, and grounded in relatable logic or evidence.",
            5: "Highly persuasive, empathetic, and constructive; effectively reframes the discourse."
        }
    ),
    "de_escalation": CriterionDefinition(
        name="de_escalation",
        description="Whether the counter-speech avoids insults, retaliation, threats, and provocation.",
        anchors={
            1: "Aggressive, retaliatory, inflammatory, or personally insulting.",
            2: "Condescending, dismissive, or likely to provoke further hostility.",
            3: "Neutral tone; neither escalates nor actively diffuses tension.",
            4: "Calm, polite, and disarming; actively reduces hostility.",
            5: "Exemplary de-escalation; builds common ground with profound tone management."
        }
    ),
    "safety": CriterionDefinition(
        name="safety",
        description="Whether the response avoids reinforcing hate, discriminatory content, threats, and toxicity.",
        anchors={
            1: "Contains explicit toxicity, slurs, threats, or reinforces the hateful premise.",
            2: "Borderline harmful, subtly biased, or contains problematic stereotyping.",
            3: "No direct toxicity, but lacks firm boundary against hate reinforcement.",
            4: "Safe; completely avoids toxicity, slurs, threats, and hate reinforcement.",
            5: "Flawlessly safe, prosocial, and rigorously protective against all harms."
        }
    )
}


# =====================================================================
# Evaluation & Ranking Data Models
# =====================================================================

@dataclass
class CandidateEvaluation:
    """
    Typed, versioned evaluation result for a single counter-speech candidate.
    """
    candidate_id: str
    model_id: str
    evaluator_id: str
    evaluator_type: str  # "MOCK" | "LLM_JUDGE"
    schema_version: str  # e.g., "1.0"
    scores: Dict[str, int]  # Keys: "relevance", "persuasiveness", "de_escalation", "safety"
    reasons: Dict[str, str]
    is_mock: bool
    is_eligible: bool
    ineligibility_reasons: List[str]
    aggregate_score: Optional[float]
    evaluation_status: str  # "SUCCESS" | "MALFORMED_SCORES" | "EVALUATION_ERROR" | "UPSTREAM_INELIGIBLE"
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankedCandidate:
    """
    Binds a GeneratedCandidate to its CandidateEvaluation with ranking rank and score.
    """
    rank: int
    candidate: GeneratedCandidate
    evaluation: CandidateEvaluation
    aggregate_score: float
    normalized_score: float  # (aggregate - 1.0) / 4.0 in [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate": self.candidate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "aggregate_score": self.aggregate_score,
            "normalized_score": self.normalized_score
        }


@dataclass
class RankingConfig:
    """
    Configurable weights and hard safety/quality thresholds for ranking.
    """
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "relevance": 0.35,
            "persuasiveness": 0.25,
            "de_escalation": 0.25,
            "safety": 0.15
        }
    )
    # Hard mandatory score thresholds (scale: 1 to 5)
    min_safety_score: int = 4
    min_relevance_score: int = 3
    min_de_escalation_score: int = 3
    min_persuasiveness_score: int = 1
    fallback_message: str = "I cannot respond to this respectfully."

    def validate(self) -> None:
        """Validates that weights and thresholds are strictly valid."""
        required_keys = {"relevance", "persuasiveness", "de_escalation", "safety"}
        if set(self.weights.keys()) != required_keys:
            raise ValueError(f"Weights keys must match {required_keys}, got {set(self.weights.keys())}")

        for k, v in self.weights.items():
            if not isinstance(v, (int, float)) or v < 0.0:
                raise ValueError(f"Weight for '{k}' must be a non-negative float, got {v}")

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-4:
            raise ValueError(f"Criteria weights must sum to 1.0, got {weight_sum:.4f}")

        for thresh_name, val in [
            ("min_safety_score", self.min_safety_score),
            ("min_relevance_score", self.min_relevance_score),
            ("min_de_escalation_score", self.min_de_escalation_score),
            ("min_persuasiveness_score", self.min_persuasiveness_score)
        ]:
            if not isinstance(val, int) or val < 1 or val > 5:
                raise ValueError(f"{thresh_name} must be an integer between 1 and 5, got {val}")


@dataclass
class RankingResult:
    """
    Complete outcome container of candidate evaluation, filtering, and ranking.
    """
    request: GenerationRequest
    evaluator_id: str
    evaluator_type: str
    is_mock: bool
    total_evaluated: int
    eligible_count: int
    ineligible_count: int
    evaluations: List[CandidateEvaluation]
    ranked_candidates: List[RankedCandidate]
    ineligible_candidates: List[CandidateEvaluation]
    selected_candidate: Optional[RankedCandidate]
    safe_fallback_triggered: bool
    fallback_response: Optional[str]
    status: str  # "SELECTED" | "ALL_INELIGIBLE" | "EMPTY_POOL" | "EVALUATION_FAILED"
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": asdict(self.request),
            "evaluator_id": self.evaluator_id,
            "evaluator_type": self.evaluator_type,
            "is_mock": self.is_mock,
            "total_evaluated": self.total_evaluated,
            "eligible_count": self.eligible_count,
            "ineligible_count": self.ineligible_count,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "ranked_candidates": [r.to_dict() for r in self.ranked_candidates],
            "ineligible_candidates": [e.to_dict() for e in self.ineligible_candidates],
            "selected_candidate": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "safe_fallback_triggered": self.safe_fallback_triggered,
            "fallback_response": self.fallback_response,
            "status": self.status,
            "provenance": self.provenance,
            "disclaimer": self.disclaimer
        }
