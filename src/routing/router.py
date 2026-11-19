"""
CSHSM Milestone 8: Fallback Routing & Threshold Calibration Infrastructure.

Provides:
- RoutingDecision: Dataclass capturing routing action (RETRIEVE vs FALLBACK_GENERATION),
  threshold applied, similarity scores, retrieved candidate or fallback payload,
  retriever provenance, and unapproved-candidate disclaimers.
- FallbackRouter: Production-grade router wrapping M7 CSHSMRetriever without modifying
  retriever internals. Evaluates top-1 similarity against an explicit calibrated threshold,
  handling exceptions, empty matches, and out-of-boundary conditions fail-closed.
- calculate_calibration_metrics: Pure statistical utility computing precision, coverage,
  fallback rate, yield, and safety rate for a candidate threshold against validation judgments.
- optimize_threshold: Grid-search threshold calibration optimizing coverage/yield subject
  to precision and safety targets, failing closed (tau = 1.0001 -> 100% fallback) if targets
  cannot be met.
"""

import math
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from src.retrieval.retriever import CSHSMRetriever, RetrievalResult, RetrievalMatch


# =====================================================================
# Threshold Validation Helper
# =====================================================================

def validate_threshold(threshold: Any) -> float:
    """
    Validates that a similarity threshold is a finite float in [-1.0, 1.05].

    The upper bound of 1.05 accommodates fail-closed synthetic thresholds
    (e.g., 1.0001) that mathematically guarantee 100% fallback in a cosine
    similarity space bounded by [-1.0, 1.0].
    """
    if threshold is None or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"Threshold must be a numeric float, got {type(threshold).__name__}"
        )
    if math.isnan(threshold) or math.isinf(threshold):
        raise ValueError(f"Threshold cannot be NaN or Inf, got {threshold}")
    val = float(threshold)
    if val < -1.0 or val > 1.05:
        raise ValueError(f"Threshold must be in [-1.0, 1.05], got {val}")
    return val


# =====================================================================
# Routing Models
# =====================================================================

@dataclass
class RoutingDecision:
    """
    Captures the deterministic routing decision for a counter-speech query.

    Actions:
      - 'RETRIEVE': Top retrieved candidate similarity meets or exceeds threshold.
      - 'FALLBACK_GENERATION': Similarity below threshold, empty retrieval,
        retriever exception, or invalid threshold.

    Fallback Reasons:
      - 'LOW_SIMILARITY': Candidate score < threshold.
      - 'EMPTY_RETRIEVAL': Index returned 0 matches for query/filter.
      - 'RETRIEVER_ERROR': Exception thrown during retrieval or query encoding.
      - 'INVALID_THRESHOLD': Threshold parameter was malformed or out of bounds.
      - None: (When action is 'RETRIEVE').
    """
    query: Optional[str]
    action: str
    threshold: Optional[float]
    similarity_score: Optional[float]
    retrieved_candidate: Optional[RetrievalMatch]
    fallback_reason: Optional[str]
    fallback_payload: Optional[Dict[str, Any]]
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "action": self.action,
            "threshold": self.threshold,
            "similarity_score": self.similarity_score,
            "retrieved_candidate": (
                self.retrieved_candidate.to_dict()
                if self.retrieved_candidate
                else None
            ),
            "fallback_reason": self.fallback_reason,
            "fallback_payload": self.fallback_payload,
            "provenance": self.provenance,
            "disclaimer": self.disclaimer
        }


# =====================================================================
# Fallback Router
# =====================================================================

class FallbackRouter:
    """
    Milestone 8 Fallback Router.

    Consumes raw similarity scores from M7 CSHSMRetriever without modifying
    M7's threshold-free retrieval logic. Evaluates top-1 candidate score s_1
    against threshold tau:
      - If s_1 >= tau: routes to 'RETRIEVE'.
      - If s_1 < tau: routes to 'FALLBACK_GENERATION' (reason: LOW_SIMILARITY).
      - If retriever returns empty: routes to 'FALLBACK_GENERATION' (reason: EMPTY_RETRIEVAL).
      - If retriever raises error: routes to 'FALLBACK_GENERATION' (reason: RETRIEVER_ERROR).

    Guarantees:
      - Never crashes or leaks raw exceptions.
      - Preserves upstream candidate disclaimers and review statuses.
      - Maintains full cryptographic and pipeline provenance.
    """

    def __init__(
        self,
        retriever: CSHSMRetriever,
        default_threshold: float = 0.80
    ):
        if retriever is None:
            raise ValueError("retriever cannot be None")
        self.retriever = retriever
        self.default_threshold = validate_threshold(default_threshold)

    def route_vector(
        self,
        query_vector: List[float],
        threshold: Optional[float] = None,
        category_filter: Optional[str] = None,
        top_k: int = 1
    ) -> RoutingDecision:
        """
        Routes a query given its pre-computed dense embedding vector.
        Fails closed on any retriever error or invalid parameter.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Validate threshold
        try:
            eff_threshold = (
                validate_threshold(threshold)
                if threshold is not None
                else self.default_threshold
            )
        except (ValueError, TypeError) as te:
            return RoutingDecision(
                query=None,
                action="FALLBACK_GENERATION",
                threshold=self.default_threshold,
                similarity_score=None,
                retrieved_candidate=None,
                fallback_reason="INVALID_THRESHOLD",
                fallback_payload={
                    "error": str(te),
                    "attempted_threshold": str(threshold)
                },
                provenance={
                    "router": "FallbackRouter",
                    "timestamp": now_iso,
                    "error": str(te)
                },
                disclaimer="SAFE FALLBACK — Invalid similarity threshold supplied; defaulting to generation."
            )

        # 2. Invoke retriever safely
        try:
            retrieval_res = self.retriever.retrieve_by_vector(
                query_vector=query_vector,
                top_k=top_k,
                category_filter=category_filter
            )
        except Exception as exc:
            return RoutingDecision(
                query=None,
                action="FALLBACK_GENERATION",
                threshold=eff_threshold,
                similarity_score=None,
                retrieved_candidate=None,
                fallback_reason="RETRIEVER_ERROR",
                fallback_payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                },
                provenance={
                    "router": "FallbackRouter",
                    "timestamp": now_iso,
                    "error": str(exc)
                },
                disclaimer="SAFE FALLBACK — Retriever exception encountered; defaulting to generation."
            )

        # 3. Evaluate outcome
        return self._evaluate_retrieval(retrieval_res, eff_threshold, query_desc=None)

    def route_text(
        self,
        query_text: str,
        threshold: Optional[float] = None,
        category_filter: Optional[str] = None,
        top_k: int = 1
    ) -> RoutingDecision:
        """
        Routes a query given raw query text. Uses configured embedder on retriever.
        Fails closed on any embedder error, retriever error, or empty text.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Validate threshold
        try:
            eff_threshold = (
                validate_threshold(threshold)
                if threshold is not None
                else self.default_threshold
            )
        except (ValueError, TypeError) as te:
            return RoutingDecision(
                query=query_text,
                action="FALLBACK_GENERATION",
                threshold=self.default_threshold,
                similarity_score=None,
                retrieved_candidate=None,
                fallback_reason="INVALID_THRESHOLD",
                fallback_payload={
                    "error": str(te),
                    "attempted_threshold": str(threshold)
                },
                provenance={
                    "router": "FallbackRouter",
                    "timestamp": now_iso,
                    "error": str(te)
                },
                disclaimer="SAFE FALLBACK — Invalid similarity threshold supplied; defaulting to generation."
            )

        # 2. Invoke retriever safely
        try:
            retrieval_res = self.retriever.retrieve_by_text(
                query_text=query_text,
                top_k=top_k,
                category_filter=category_filter
            )
        except Exception as exc:
            return RoutingDecision(
                query=query_text,
                action="FALLBACK_GENERATION",
                threshold=eff_threshold,
                similarity_score=None,
                retrieved_candidate=None,
                fallback_reason="RETRIEVER_ERROR",
                fallback_payload={
                    "query": query_text,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                },
                provenance={
                    "router": "FallbackRouter",
                    "timestamp": now_iso,
                    "error": str(exc)
                },
                disclaimer="SAFE FALLBACK — Retriever exception encountered; defaulting to generation."
            )

        # 3. Evaluate outcome
        return self._evaluate_retrieval(retrieval_res, eff_threshold, query_desc=query_text)

    def _evaluate_retrieval(
        self,
        retrieval_res: RetrievalResult,
        eff_threshold: float,
        query_desc: Optional[str] = None
    ) -> RoutingDecision:
        """Evaluates top candidate against threshold with full provenance preservation."""
        query_identifier = query_desc or retrieval_res.query_text or retrieval_res.query_vector_sha256
        provenance = dict(retrieval_res.provenance)
        provenance["threshold_applied"] = eff_threshold

        # Case A: Empty matches
        if not retrieval_res.matches:
            provenance["routing_action"] = "FALLBACK_GENERATION"
            provenance["fallback_reason"] = "EMPTY_RETRIEVAL"
            return RoutingDecision(
                query=query_identifier,
                action="FALLBACK_GENERATION",
                threshold=eff_threshold,
                similarity_score=None,
                retrieved_candidate=None,
                fallback_reason="EMPTY_RETRIEVAL",
                fallback_payload={
                    "query": query_identifier,
                    "retrieval_status": retrieval_res.retrieval_status,
                    "matches_count": 0
                },
                provenance=provenance,
                disclaimer="SAFE FALLBACK — No matching counter-speech candidates found; defaulting to generation."
            )

        top_match = retrieval_res.matches[0]
        top_score = top_match.score

        # Case B: Top candidate meets or exceeds threshold -> RETRIEVE
        if top_score >= eff_threshold:
            provenance["routing_action"] = "RETRIEVE"
            provenance["similarity_score"] = top_score
            return RoutingDecision(
                query=query_identifier,
                action="RETRIEVE",
                threshold=eff_threshold,
                similarity_score=top_score,
                retrieved_candidate=top_match,
                fallback_reason=None,
                fallback_payload=None,
                provenance=provenance,
                disclaimer=retrieval_res.disclaimer
            )

        # Case C: Top candidate similarity below threshold -> FALLBACK_GENERATION
        provenance["routing_action"] = "FALLBACK_GENERATION"
        provenance["fallback_reason"] = "LOW_SIMILARITY"
        provenance["similarity_score"] = top_score
        return RoutingDecision(
            query=query_identifier,
            action="FALLBACK_GENERATION",
            threshold=eff_threshold,
            similarity_score=top_score,
            retrieved_candidate=None,
            fallback_reason="LOW_SIMILARITY",
            fallback_payload={
                "query": query_identifier,
                "top_score": top_score,
                "threshold": eff_threshold,
                "top_candidate": top_match.to_dict()
            },
            provenance=provenance,
            disclaimer="SAFE FALLBACK — Top candidate similarity below threshold; defaulting to generation."
        )


# =====================================================================
# Calibration Metrics & Threshold Optimization
# =====================================================================

def calculate_calibration_metrics(
    scores: List[float],
    labels: List[int],
    safety_labels: Optional[List[int]] = None,
    threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Computes retrieval-versus-generation routing calibration metrics for a candidate threshold.

    Definitions:
      - Coverage(tau) = N_retrieved / N_total
      - FallbackRate(tau) = N_fallback / N_total = 1.0 - Coverage(tau)
      - Precision(tau) = N_relevant_retrieved / N_retrieved (None if N_retrieved == 0)
      - SafetyRate(tau) = N_safe_retrieved / N_retrieved (None if N_retrieved == 0 or safety_labels is None)
      - Yield(tau) = N_relevant_retrieved / N_total (0.0 if N_retrieved == 0 or N_total == 0)

    Edge Cases:
      - When N_retrieved == 0: Precision and SafetyRate are undefined (None), NOT 1.0 or 0.0.
      - When N_total == 0: Coverage, Yield are 0.0, FallbackRate is 1.0, Precision/Safety are None.
    """
    eff_threshold = validate_threshold(threshold)

    if len(scores) != len(labels):
        raise ValueError(
            f"Dimension mismatch: scores ({len(scores)}) != labels ({len(labels)})"
        )
    if safety_labels is not None and len(safety_labels) != len(scores):
        raise ValueError(
            f"Dimension mismatch: safety_labels ({len(safety_labels)}) != scores ({len(scores)})"
        )

    for i, s in enumerate(scores):
        if not isinstance(s, (int, float)) or isinstance(s, bool) or math.isnan(s) or math.isinf(s):
            raise ValueError(f"Score at index {i} is not a finite numeric value: {s}")

    for i, y in enumerate(labels):
        if y not in (0, 1):
            raise ValueError(f"Label at index {i} must be binary 0 or 1, got: {y}")

    if safety_labels is not None:
        for i, s_lab in enumerate(safety_labels):
            if s_lab not in (0, 1):
                raise ValueError(f"Safety label at index {i} must be binary 0 or 1, got: {s_lab}")

    n_total = len(scores)
    if n_total == 0:
        return {
            "threshold": eff_threshold,
            "n_total": 0,
            "n_retrieved": 0,
            "n_fallback": 0,
            "n_relevant_retrieved": 0,
            "n_safe_retrieved": 0 if safety_labels is not None else None,
            "coverage": 0.0,
            "fallback_rate": 1.0,
            "precision": None,
            "safety_rate": None,
            "yield": 0.0
        }

    # Identify retrieved queries where score >= threshold
    retrieved_indices = [i for i, s in enumerate(scores) if s >= eff_threshold]
    n_retrieved = len(retrieved_indices)
    n_fallback = n_total - n_retrieved

    coverage = round(n_retrieved / n_total, 6)
    fallback_rate = round(n_fallback / n_total, 6)

    if n_retrieved > 0:
        relevant_count = sum(labels[i] for i in retrieved_indices)
        precision = round(relevant_count / n_retrieved, 6)
        retrieval_yield = round(relevant_count / n_total, 6)

        if safety_labels is not None:
            safe_count = sum(safety_labels[i] for i in retrieved_indices)
            safety_rate = round(safe_count / n_retrieved, 6)
        else:
            safe_count = None
            safety_rate = None
    else:
        # Precision and safety are undefined (None) when retrieved count is zero
        relevant_count = 0
        precision = None
        safety_rate = None
        safe_count = 0 if safety_labels is not None else None
        retrieval_yield = 0.0

    return {
        "threshold": eff_threshold,
        "n_total": n_total,
        "n_retrieved": n_retrieved,
        "n_fallback": n_fallback,
        "n_relevant_retrieved": relevant_count,
        "n_safe_retrieved": safe_count,
        "coverage": coverage,
        "fallback_rate": fallback_rate,
        "precision": precision,
        "safety_rate": safety_rate,
        "yield": retrieval_yield
    }


def optimize_threshold(
    scores: List[float],
    labels: List[int],
    safety_labels: Optional[List[int]] = None,
    min_threshold: float = 0.50,
    max_threshold: float = 0.95,
    step: float = 0.01,
    target_precision: float = 0.85,
    required_safety: float = 1.00,
    objective: str = "coverage"
) -> Dict[str, Any]:
    """
    Optimizes the routing similarity threshold via deterministic grid search.

    Objective:
      Maximizes coverage (or yield) subject to:
        1. precision(tau) >= target_precision
        2. safety_rate(tau) >= required_safety (if safety_labels provided)
        3. N_retrieved > 0

    Tie-breaking:
      When multiple candidate thresholds yield identical primary metric (coverage or yield):
        - Secondary metric: precision (descending)
        - Tertiary metric: yield (descending)
        - Quaternary tie-breaker: threshold (descending, more conservative against over-retrieval)

    Fail-Closed Safety:
      If no candidate threshold satisfies the constraints (or if data is empty),
      the optimizer selects tau* = 1.0001, forcing 100% fallback to generation
      in the router's [-1.0, 1.0] similarity domain.
    """
    if min_threshold > max_threshold:
        raise ValueError(
            f"min_threshold ({min_threshold}) cannot exceed max_threshold ({max_threshold})"
        )
    if step <= 0.0:
        raise ValueError(f"step must be strictly positive, got {step}")
    if not (0.0 <= target_precision <= 1.0):
        raise ValueError(f"target_precision must be in [0.0, 1.0], got {target_precision}")
    if not (0.0 <= required_safety <= 1.0):
        raise ValueError(f"required_safety must be in [0.0, 1.0], got {required_safety}")
    if objective not in ("coverage", "yield"):
        raise ValueError(f"objective must be 'coverage' or 'yield', got {objective}")

    # Generate candidate grid
    num_steps = int(round((max_threshold - min_threshold) / step)) + 1
    grid = [round(min_threshold + i * step, 6) for i in range(num_steps)]

    evaluated = []
    feasible = []

    for tau in grid:
        m = calculate_calibration_metrics(
            scores=scores,
            labels=labels,
            safety_labels=safety_labels,
            threshold=tau
        )
        evaluated.append(m)

        # Check feasibility
        if m["n_retrieved"] == 0 or m["precision"] is None:
            continue
        if m["precision"] < target_precision:
            continue
        if safety_labels is not None:
            if m["safety_rate"] is None or m["safety_rate"] < required_safety:
                continue

        feasible.append(m)

    # Fail closed if no threshold satisfies constraints
    if not feasible:
        fail_closed_tau = 1.0001
        fail_closed_metrics = calculate_calibration_metrics(
            scores=scores,
            labels=labels,
            safety_labels=safety_labels,
            threshold=fail_closed_tau
        )
        return {
            "optimal_threshold": fail_closed_tau,
            "status": "FAILED_CONSTRAINTS_FAIL_CLOSED",
            "target_precision": target_precision,
            "required_safety": required_safety,
            "achieved_metrics": fail_closed_metrics,
            "objective": objective,
            "grid_search_summary": {
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
                "step": step,
                "total_evaluated": len(grid),
                "feasible_count": 0
            },
            "disclaimer": (
                "CALIBRATION PROTOCOL ONLY — Uncalibrated on real data. "
                "Pending Tier-2 human or Tier-3 LLM acceptability annotations."
            )
        }

    # Deterministic sorting with conservative tie-breaking
    if objective == "coverage":
        # Sort by: coverage (desc), precision (desc), yield (desc), threshold (desc)
        feasible.sort(
            key=lambda m: (m["coverage"], m["precision"], m["yield"], m["threshold"]),
            reverse=True
        )
    else:
        # Sort by: yield (desc), coverage (desc), precision (desc), threshold (desc)
        feasible.sort(
            key=lambda m: (m["yield"], m["coverage"], m["precision"], m["threshold"]),
            reverse=True
        )

    best_m = feasible[0]
    return {
        "optimal_threshold": best_m["threshold"],
        "status": "OPTIMAL",
        "target_precision": target_precision,
        "required_safety": required_safety,
        "achieved_metrics": best_m,
        "objective": objective,
        "grid_search_summary": {
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
            "step": step,
            "total_evaluated": len(grid),
            "feasible_count": len(feasible)
        },
        "disclaimer": (
            "CALIBRATION PROTOCOL ONLY — Uncalibrated on real data. "
            "Pending Tier-2 human or Tier-3 LLM acceptability annotations."
        )
    }
