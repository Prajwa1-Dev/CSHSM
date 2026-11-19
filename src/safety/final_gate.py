"""
CSHSM Milestone 11: Final Safety & Validation Gate.

Provides:
- FinalGateResult: Typed, versioned outcome container for the final safety boundary.
- FinalSafetyGate: Independent gate that checks the selected candidate from M10
  before any response is eligible for delivery.
- validate_final_candidate: Convenience functional interface for final gate validation.

Guarantees:
- Candidate integrity (non-empty text, valid ID, identity consistency with M10).
- Upstream validation audit (M9 structural, M9 safety, M10 evaluation success, M10 gates).
- Defense-in-depth: Re-evaluates deterministic safety filter on the candidate text.
- Fail-closed behavior: If any check fails, or if selected candidate is missing,
  triggers safe fallback ("I cannot respond to this respectfully.").
- Explicit mock/development mode guard: When allow_mock=False, simulated evidence is rejected.
- Invariants: Never automatically approves candidates; maintains review_status="unreviewed"
  and quality_status="generated_candidate".
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json

from src.kb.filters import safety_filter
from src.generation.fan_out import GeneratedCandidate
from src.evaluation.schema import RankedCandidate, RankingResult, CandidateEvaluation
from src.evaluation.evaluator import validate_scores


# =====================================================================
# Result Model
# =====================================================================

@dataclass
class FinalGateResult:
    """
    Versioned, auditable outcome of the M11 final safety and validation gate.

    Decisions:
      - "PASS_UNAPPROVED": Candidate passed all structural, safety, and evaluation gates
        in explicit development/test mode (or real mode). Retains unapproved status.
      - "REJECTED": One or more mandatory integrity, upstream, or safety checks failed.
      - "FALLBACK": Upstream fallback was signaled, or no eligible candidate existed.
    """
    decision: str  # "PASS_UNAPPROVED" | "REJECTED" | "FALLBACK"
    candidate_id: Optional[str]
    text: Optional[str]
    model_id: Optional[str]
    provider: Optional[str]
    is_mock_generation: bool
    is_mock_evaluation: bool
    evaluation_scores: Optional[Dict[str, int]]
    aggregate_score: Optional[float]
    checks: Dict[str, Dict[str, Any]]
    failure_reasons: List[str]
    safe_fallback_triggered: bool
    fallback_response: Optional[str]
    quality_status: str  # "generated_candidate" or "fallback"
    review_status: str   # "unreviewed"
    human_audited: bool  # Always False
    schema_version: str  # "1.0"
    timestamp: str
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Final Safety & Validation Gate
# =====================================================================

class FinalSafetyGate:
    """
    Milestone 11 Final Safety & Validation Gate.

    Acts as an independent safety boundary between M10 evaluation/ranking
    and downstream delivery. Does not trust candidates solely because M10
    selected or ranked them highly.
    """

    DEFAULT_FALLBACK_MESSAGE = "I cannot respond to this respectfully."
    SCHEMA_VERSION = "1.0"
    DISCLAIMER_UNAPPROVED = (
        "VALIDATED UNAPPROVED CANDIDATE — Passed M11 final safety and validation gate. "
        "Unapproved for production delivery pending final human review and release policy."
    )
    DISCLAIMER_UNAPPROVED_RETRIEVED = (
        "VALIDATED UNAPPROVED RETRIEVED CANDIDATE — Passed M11 final safety and validation gate. "
        "Unapproved for production delivery pending final human review and release policy."
    )
    DISCLAIMER_FALLBACK = (
        "SAFE FALLBACK — Pipeline triggered safe fallback; candidate rejected or unavailable."
    )

    def __init__(
        self,
        fallback_message: str = DEFAULT_FALLBACK_MESSAGE,
        min_safety_score: int = 4,
        min_relevance_score: int = 3,
        min_de_escalation_score: int = 3,
        min_persuasiveness_score: int = 1
    ):
        self.fallback_message = fallback_message
        self.min_safety_score = min_safety_score
        self.min_relevance_score = min_relevance_score
        self.min_de_escalation_score = min_de_escalation_score
        self.min_persuasiveness_score = min_persuasiveness_score

    def validate(
        self,
        ranking_result_or_candidate: Union[RankingResult, RankedCandidate],
        allow_mock: bool = True
    ) -> FinalGateResult:
        """
        Validates the candidate through all safety and integrity gates.

        Parameters:
          - ranking_result_or_candidate: Either an M10 RankingResult or RankedCandidate.
          - allow_mock: If True, allows simulated/mock evidence with explicit labeling.
            If False (production mode), rejects simulated evidence fail-closed.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Handle RankingResult input
        if isinstance(ranking_result_or_candidate, RankingResult):
            ranking_result = ranking_result_or_candidate

            # Check if upstream fallback was triggered or pool was empty
            if ranking_result.safe_fallback_triggered or not ranking_result.selected_candidate:
                return self._create_fallback_result(
                    reason=f"Upstream fallback triggered: status={ranking_result.status}",
                    now_iso=now_iso,
                    provenance=ranking_result.provenance
                )
            ranked_cand = ranking_result.selected_candidate
            ranking_provenance = ranking_result.provenance
        elif isinstance(ranking_result_or_candidate, RankedCandidate):
            ranked_cand = ranking_result_or_candidate
            ranking_provenance = {}
        else:
            raise TypeError(
                f"Expected RankingResult or RankedCandidate, got {type(ranking_result_or_candidate).__name__}"
            )

        cand = ranked_cand.candidate
        eval_res = ranked_cand.evaluation

        checks: Dict[str, Dict[str, Any]] = {}
        failure_reasons: List[str] = []

        # =================================================================
        # Check A: Candidate Integrity
        # =================================================================
        integrity_ok = True
        integrity_reasons = []

        if not cand:
            integrity_ok = False
            integrity_reasons.append("Candidate object is None.")
        else:
            if not cand.text or not cand.text.strip():
                integrity_ok = False
                integrity_reasons.append("Candidate text is empty or whitespace.")

            if not cand.candidate_id or not cand.candidate_id.strip():
                integrity_ok = False
                integrity_reasons.append("Candidate candidate_id is missing or empty.")

            if not cand.model_id or not cand.model_id.strip():
                integrity_ok = False
                integrity_reasons.append("Candidate model_id is missing or empty.")

            # Cross-check consistency with M10 evaluation record
            if eval_res:
                if cand.candidate_id != eval_res.candidate_id:
                    integrity_ok = False
                    integrity_reasons.append(
                        f"Candidate ID mismatch: M9 candidate ({cand.candidate_id}) "
                        f"!= M10 evaluation ({eval_res.candidate_id})."
                    )
                if cand.model_id != eval_res.model_id:
                    integrity_ok = False
                    integrity_reasons.append(
                        f"Model ID mismatch: M9 candidate ({cand.model_id}) "
                        f"!= M10 evaluation ({eval_res.model_id})."
                    )

        checks["candidate_integrity"] = {
            "passed": integrity_ok,
            "reasons": integrity_reasons
        }
        if not integrity_ok:
            failure_reasons.extend(integrity_reasons)

        # =================================================================
        # Check B: Upstream M9 Validation Verification
        # =================================================================
        m9_struct_ok = True
        m9_safety_ok = True
        upstream_reasons = []

        if not cand or not isinstance(cand.structural_validation, dict):
            m9_struct_ok = False
            upstream_reasons.append("Missing or malformed M9 structural_validation dictionary.")
        elif not cand.structural_validation.get("is_valid", False):
            m9_struct_ok = False
            val_reason = cand.structural_validation.get("validation_reason", "unknown")
            upstream_reasons.append(f"M9 structural validation failed: {val_reason}.")

        if not cand or not isinstance(cand.safety_validation, dict):
            m9_safety_ok = False
            upstream_reasons.append("Missing or malformed M9 safety_validation dictionary.")
        elif not cand.safety_validation.get("is_safe", False):
            m9_safety_ok = False
            rej_reason = cand.safety_validation.get("rejection_reason", "unknown")
            upstream_reasons.append(f"M9 safety validation failed: {rej_reason}.")

        checks["m9_structural_validation"] = {
            "passed": m9_struct_ok,
            "details": cand.structural_validation if cand else {}
        }
        checks["m9_safety_validation"] = {
            "passed": m9_safety_ok,
            "details": cand.safety_validation if cand else {}
        }
        if not m9_struct_ok or not m9_safety_ok:
            failure_reasons.extend(upstream_reasons)

        # =================================================================
        # Check C: Upstream M10 Evaluation & Mandatory Gates
        # =================================================================
        m10_ok = True
        m10_reasons = []

        if not eval_res:
            m10_ok = False
            m10_reasons.append("Missing M10 CandidateEvaluation object.")
        else:
            if eval_res.evaluation_status != "SUCCESS":
                m10_ok = False
                m10_reasons.append(f"M10 evaluation status is not SUCCESS ({eval_res.evaluation_status}).")

            if not eval_res.is_eligible:
                m10_ok = False
                m10_reasons.append(f"M10 marked candidate ineligible: {eval_res.ineligibility_reasons}.")

            is_valid_scores, score_err, clean_scores = validate_scores(eval_res.scores)
            if not is_valid_scores:
                m10_ok = False
                m10_reasons.append(f"M10 scores validation failed: {score_err}.")
            else:
                # Re-verify hard thresholds independently
                safety_score = clean_scores.get("safety", 0)
                relevance_score = clean_scores.get("relevance", 0)
                deesc_score = clean_scores.get("de_escalation", 0)
                pers_score = clean_scores.get("persuasiveness", 0)

                if safety_score < self.min_safety_score:
                    m10_ok = False
                    m10_reasons.append(
                        f"Safety score ({safety_score}) below final gate minimum ({self.min_safety_score})."
                    )
                if relevance_score < self.min_relevance_score:
                    m10_ok = False
                    m10_reasons.append(
                        f"Relevance score ({relevance_score}) below final gate minimum ({self.min_relevance_score})."
                    )
                if deesc_score < self.min_de_escalation_score:
                    m10_ok = False
                    m10_reasons.append(
                        f"De-escalation score ({deesc_score}) below final gate minimum ({self.min_de_escalation_score})."
                    )
                if pers_score < self.min_persuasiveness_score:
                    m10_ok = False
                    m10_reasons.append(
                        f"Persuasiveness score ({pers_score}) below final gate minimum ({self.min_persuasiveness_score})."
                    )

        checks["m10_evaluation_gates"] = {
            "passed": m10_ok,
            "reasons": m10_reasons
        }
        if not m10_ok:
            failure_reasons.extend(m10_reasons)

        # =================================================================
        # Check D: Mock Evidence / Environment Mode Policy
        # =================================================================
        is_mock_gen = bool(cand.is_mock) if cand else False
        is_mock_eval = bool(eval_res.is_mock or eval_res.evaluator_type == "MOCK") if eval_res else False
        mock_mode_ok = True
        mock_reasons = []

        if (is_mock_gen or is_mock_eval) and not allow_mock:
            mock_mode_ok = False
            mock_reasons.append(
                "Mock or simulated generation/evaluation evidence is prohibited when allow_mock=False (production mode)."
            )

        checks["mock_evidence_policy"] = {
            "passed": mock_mode_ok,
            "allow_mock": allow_mock,
            "is_mock_generation": is_mock_gen,
            "is_mock_evaluation": is_mock_eval,
            "reasons": mock_reasons
        }
        if not mock_mode_ok:
            failure_reasons.extend(mock_reasons)

        # =================================================================
        # Check E: Independent Defense-in-Depth Final Safety Filter
        # =================================================================
        final_safety_ok = True
        final_safety_reasons = []

        if cand and cand.text and cand.text.strip():
            filter_safe, rej_reason = safety_filter(cand.text.strip())
            if not filter_safe:
                final_safety_ok = False
                final_safety_reasons.append(f"Independent final safety filter rejected text: '{rej_reason}'.")
        else:
            final_safety_ok = False
            final_safety_reasons.append("Cannot apply safety filter: candidate text is empty.")

        checks["final_safety_filter"] = {
            "passed": final_safety_ok,
            "reasons": final_safety_reasons
        }
        if not final_safety_ok:
            failure_reasons.extend(final_safety_reasons)

        # =================================================================
        # Final Decision Synthesis
        # =================================================================
        all_passed = (
            integrity_ok and
            m9_struct_ok and
            m9_safety_ok and
            m10_ok and
            mock_mode_ok and
            final_safety_ok
        )

        provenance = {
            "gate_version": self.SCHEMA_VERSION,
            "timestamp": now_iso,
            "allow_mock": allow_mock,
            "is_mock_generation": is_mock_gen,
            "is_mock_evaluation": is_mock_eval,
            "upstream_provenance": ranking_provenance
        }

        if all_passed:
            return FinalGateResult(
                decision="PASS_UNAPPROVED",
                candidate_id=cand.candidate_id,
                text=cand.text.strip(),
                model_id=cand.model_id,
                provider=cand.provider,
                is_mock_generation=is_mock_gen,
                is_mock_evaluation=is_mock_eval,
                evaluation_scores=eval_res.scores if eval_res else None,
                aggregate_score=ranked_cand.aggregate_score,
                checks=checks,
                failure_reasons=[],
                safe_fallback_triggered=False,
                fallback_response=None,
                quality_status="generated_candidate",
                review_status="unreviewed",
                human_audited=False,
                schema_version=self.SCHEMA_VERSION,
                timestamp=now_iso,
                provenance=provenance,
                disclaimer=self.DISCLAIMER_UNAPPROVED
            )
        else:
            return FinalGateResult(
                decision="REJECTED",
                candidate_id=cand.candidate_id if cand else None,
                text=None,
                model_id=cand.model_id if cand else None,
                provider=cand.provider if cand else None,
                is_mock_generation=is_mock_gen,
                is_mock_evaluation=is_mock_eval,
                evaluation_scores=eval_res.scores if eval_res else None,
                aggregate_score=ranked_cand.aggregate_score,
                checks=checks,
                failure_reasons=failure_reasons,
                safe_fallback_triggered=True,
                fallback_response=self.fallback_message,
                quality_status="fallback",
                review_status="unreviewed",
                human_audited=False,
                schema_version=self.SCHEMA_VERSION,
                timestamp=now_iso,
                provenance=provenance,
                disclaimer=self.DISCLAIMER_FALLBACK
            )

    def _create_fallback_result(
        self,
        reason: str,
        now_iso: str,
        provenance: Dict[str, Any]
    ) -> FinalGateResult:
        """Constructs a deterministic fallback result when upstream yields no candidate."""
        return FinalGateResult(
            decision="FALLBACK",
            candidate_id=None,
            text=None,
            model_id=None,
            provider=None,
            is_mock_generation=True,
            is_mock_evaluation=True,
            evaluation_scores=None,
            aggregate_score=None,
            checks={
                "upstream_selection": {
                    "passed": False,
                    "reason": reason
                }
            },
            failure_reasons=[reason],
            safe_fallback_triggered=True,
            fallback_response=self.fallback_message,
            quality_status="fallback",
            review_status="unreviewed",
            human_audited=False,
            schema_version=self.SCHEMA_VERSION,
            timestamp=now_iso,
            provenance={
                "gate_version": self.SCHEMA_VERSION,
                "timestamp": now_iso,
                "upstream_provenance": provenance
            },
            disclaimer=self.DISCLAIMER_FALLBACK
        )

    def validate_retrieved_candidate(
        self,
        retrieval_match: Any,
        routing_decision: Optional[Any] = None,
        allow_mock: bool = True
    ) -> FinalGateResult:
        """
        Validates a retrieved candidate from M7/M8 through final safety and integrity gates.

        Ensures:
          - Non-empty text and valid kb_id.
          - Consistency with routing decision (if provided, must have action == "RETRIEVE").
          - Mock evidence policy against allow_mock.
          - Independent defense-in-depth safety filter application.
          - Invariants: Never marks as generated_candidate or approved; preserves unreviewed status.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        checks: Dict[str, Dict[str, Any]] = {}
        failure_reasons: List[str] = []

        # 1. Candidate Integrity
        integrity_ok = True
        integrity_reasons = []
        if not retrieval_match:
            integrity_ok = False
            integrity_reasons.append("Retrieved match candidate is None.")
            cand_id = None
            cand_text = None
            source_provider = "knowledge_base"
            score = None
            quality = "fallback"
        else:
            cand_id = getattr(retrieval_match, "kb_id", None)
            cand_text = getattr(retrieval_match, "counter_speech", None)
            source_provider = getattr(retrieval_match, "source", None) or "knowledge_base"
            score = getattr(retrieval_match, "score", None)
            quality = getattr(retrieval_match, "quality_status", "filtered_candidate")

            # Prevent false approval
            if quality == "approved":
                quality = "filtered_candidate"

            if not cand_id or not str(cand_id).strip():
                integrity_ok = False
                integrity_reasons.append("Retrieved candidate kb_id is missing or empty.")

            if not cand_text or not str(cand_text).strip():
                integrity_ok = False
                integrity_reasons.append("Retrieved candidate counter_speech text is empty.")

        checks["retrieved_integrity"] = {
            "passed": integrity_ok,
            "reasons": integrity_reasons
        }
        if not integrity_ok:
            failure_reasons.extend(integrity_reasons)

        # 2. Routing Decision Consistency
        routing_ok = True
        routing_reasons = []
        if routing_decision is not None:
            action = getattr(routing_decision, "action", None)
            if action != "RETRIEVE":
                routing_ok = False
                routing_reasons.append(
                    f"Routing decision action is '{action}', expected 'RETRIEVE'."
                )
        checks["routing_consistency"] = {
            "passed": routing_ok,
            "reasons": routing_reasons
        }
        if not routing_ok:
            failure_reasons.extend(routing_reasons)

        # 3. Mock Evidence Policy
        is_mock_retrieval = False
        if routing_decision is not None:
            prov = getattr(routing_decision, "provenance", {}) or {}
            ret_prov = prov.get("retrieval_result_provenance", {}) or {}
            if prov.get("is_mock") or ret_prov.get("is_mock"):
                is_mock_retrieval = True
        if retrieval_match and getattr(retrieval_match, "is_mock", False):
            is_mock_retrieval = True

        mock_mode_ok = True
        mock_reasons = []
        if is_mock_retrieval and not allow_mock:
            mock_mode_ok = False
            mock_reasons.append(
                "Mock retrieval evidence is prohibited when allow_mock=False (production mode)."
            )
        checks["mock_evidence_policy"] = {
            "passed": mock_mode_ok,
            "allow_mock": allow_mock,
            "is_mock_retrieval": is_mock_retrieval,
            "reasons": mock_reasons
        }
        if not mock_mode_ok:
            failure_reasons.extend(mock_reasons)

        # 4. Independent Final Safety Filter
        final_safety_ok = True
        final_safety_reasons = []
        if cand_text and str(cand_text).strip():
            filter_safe, rej_reason = safety_filter(str(cand_text).strip())
            if not filter_safe:
                final_safety_ok = False
                final_safety_reasons.append(
                    f"Independent final safety filter rejected retrieved text: '{rej_reason}'."
                )
        else:
            final_safety_ok = False
            final_safety_reasons.append("Cannot apply safety filter: retrieved text is empty.")

        checks["final_safety_filter"] = {
            "passed": final_safety_ok,
            "reasons": final_safety_reasons
        }
        if not final_safety_ok:
            failure_reasons.extend(final_safety_reasons)

        # Synthesis
        all_passed = integrity_ok and routing_ok and mock_mode_ok and final_safety_ok

        provenance = {
            "gate_version": self.SCHEMA_VERSION,
            "timestamp": now_iso,
            "source_type": "retrieval",
            "allow_mock": allow_mock,
            "is_mock_retrieval": is_mock_retrieval,
            "routing_decision": (
                routing_decision.to_dict() if hasattr(routing_decision, "to_dict") else str(routing_decision)
            ) if routing_decision else None
        }

        if all_passed:
            return FinalGateResult(
                decision="PASS_UNAPPROVED",
                candidate_id=cand_id,
                text=str(cand_text).strip(),
                model_id="kb_retrieval",
                provider=source_provider,
                is_mock_generation=False,
                is_mock_evaluation=False,
                evaluation_scores=None,
                aggregate_score=score,
                checks=checks,
                failure_reasons=[],
                safe_fallback_triggered=False,
                fallback_response=None,
                quality_status=quality,
                review_status="unreviewed",
                human_audited=False,
                schema_version=self.SCHEMA_VERSION,
                timestamp=now_iso,
                provenance=provenance,
                disclaimer=self.DISCLAIMER_UNAPPROVED_RETRIEVED
            )
        else:
            return FinalGateResult(
                decision="REJECTED",
                candidate_id=cand_id,
                text=None,
                model_id="kb_retrieval",
                provider=source_provider,
                is_mock_generation=False,
                is_mock_evaluation=False,
                evaluation_scores=None,
                aggregate_score=score,
                checks=checks,
                failure_reasons=failure_reasons,
                safe_fallback_triggered=True,
                fallback_response=self.fallback_message,
                quality_status="fallback",
                review_status="unreviewed",
                human_audited=False,
                schema_version=self.SCHEMA_VERSION,
                timestamp=now_iso,
                provenance=provenance,
                disclaimer=self.DISCLAIMER_FALLBACK
            )


# =====================================================================
# Functional Helper
# =====================================================================

def validate_final_candidate(
    ranking_result_or_candidate: Union[RankingResult, RankedCandidate],
    allow_mock: bool = True,
    fallback_message: str = FinalSafetyGate.DEFAULT_FALLBACK_MESSAGE
) -> FinalGateResult:
    """
    Convenience functional interface for final safety gate validation of generated candidates.
    """
    gate = FinalSafetyGate(fallback_message=fallback_message)
    return gate.validate(ranking_result_or_candidate, allow_mock=allow_mock)


def validate_retrieved_candidate(
    retrieval_match: Any,
    routing_decision: Optional[Any] = None,
    allow_mock: bool = True,
    fallback_message: str = FinalSafetyGate.DEFAULT_FALLBACK_MESSAGE
) -> FinalGateResult:
    """
    Convenience functional interface for final safety gate validation of retrieved candidates.
    """
    gate = FinalSafetyGate(fallback_message=fallback_message)
    return gate.validate_retrieved_candidate(
        retrieval_match=retrieval_match,
        routing_decision=routing_decision,
        allow_mock=allow_mock
    )
