"""
CSHSM Milestone 10: Candidate Evaluation, Ranking & Metrics Package.
"""

try:
    from src.evaluation.metrics import generate_evaluation_report
except ImportError:
    generate_evaluation_report = None
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
    validate_scores
)
from src.evaluation.ranker import CandidateRanker

__all__ = [
    "generate_evaluation_report",
    "CriterionDefinition",
    "EVALUATION_RUBRICS",
    "CandidateEvaluation",
    "RankedCandidate",
    "RankingConfig",
    "RankingResult",
    "BaseCandidateEvaluator",
    "MockCandidateEvaluator",
    "validate_scores",
    "CandidateRanker"
]
