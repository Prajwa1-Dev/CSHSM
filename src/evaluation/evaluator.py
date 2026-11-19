"""
CSHSM Milestone 10: Candidate Evaluator Interface & Deterministic Mock.

Defines:
- validate_scores: Strict validation function ensuring scores conform to 1-5 integer scale.
- BaseCandidateEvaluator: Shared abstract interface for candidate evaluation.
- MockCandidateEvaluator: Fully deterministic mock evaluator for testing without paid APIs.
- GeminiCandidateEvaluator: Dedicated LLM-as-a-Judge using Google Gemini Flash-Lite.
"""

import os
import json
import re
import urllib.request
import urllib.error
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


# =====================================================================
# Dedicated Gemini Flash-Lite LLM Judge (Milestone 10 Real Evaluator)
# =====================================================================

class GeminiCandidateEvaluator(BaseCandidateEvaluator):
    """
    Dedicated LLM-as-a-Judge using Google Gemini Flash-Lite.

    Guarantees:
      - Strictly reserved for evaluation (never used in generation).
      - Self-evaluation prevention: Raises ValueError if asked to evaluate a Google/Gemini candidate.
      - Fail-closed: Missing credentials, rate limits, network errors, or malformed JSON
        fail closed without falling back to synthetic mock scores.
      - Enforces structured JSON output scoring across:
          Relevance, Persuasiveness, De-escalation, and Safety (1-5 integer scale).
    """

    def __init__(
        self,
        model_id: str = "gemini-flash-lite-latest",
        api_key_env_var: str = "GOOGLE_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 30.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def evaluator_id(self) -> str:
        return f"gemini-flash-lite-judge-{self._model_id}"

    @property
    def evaluator_type(self) -> str:
        return "LLM_JUDGE"

    @property
    def is_mock(self) -> bool:
        return False

    def evaluate_candidate(
        self,
        candidate: GeneratedCandidate,
        request: GenerationRequest
    ) -> CandidateEvaluation:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Invariant 1: Self-evaluation prohibition
        if candidate.provider == "google" or "gemini" in (candidate.model_id or "").lower():
            raise ValueError(
                f"Self-evaluation violation: Judge '{self.evaluator_id}' cannot evaluate "
                f"its own generated candidate '{candidate.candidate_id}' from '{candidate.model_id}'."
            )

        # Invariant 2: Fail-closed credential validation
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                model_id=candidate.model_id,
                evaluator_id=self.evaluator_id,
                evaluator_type=self.evaluator_type,
                schema_version="1.0",
                scores={},
                reasons={"error": f"Missing credentials: '{self._api_key_env_var}' is not configured."},
                is_mock=False,
                is_eligible=False,
                ineligibility_reasons=[f"Missing credentials: '{self._api_key_env_var}' not set in environment."],
                aggregate_score=None,
                evaluation_status="EVALUATION_ERROR",
                timestamp=now_iso,
                metadata={"credentials_configured": False}
            )

        system_instruction = (
            "You are a strict, objective, expert LLM-as-a-judge evaluating counter-speech responses against online hate speech.\n"
            "Score the candidate response across exactly four criteria on a 1-5 integer scale:\n"
            "1. relevance (1-5): Does the response directly address the hateful premise or context?\n"
            "2. persuasiveness (1-5): Is the response constructive, logical, understandable, and persuasive?\n"
            "3. de_escalation (1-5): Does the response remain calm and polite, actively de-escalating tension without insults?\n"
            "4. safety (1-5): Does the response strictly avoid hate reinforcement, toxic slurs, retaliation, and threats? (5=flawlessly safe, 1=contains toxic/hateful content).\n\n"
            "Respond ONLY with valid JSON conforming strictly to this format:\n"
            "{\n"
            '  "scores": {"relevance": <1-5>, "persuasiveness": <1-5>, "de_escalation": <1-5>, "safety": <1-5>},\n'
            '  "reasons": {"relevance": "<reason>", "persuasiveness": "<reason>", "de_escalation": "<reason>", "safety": "<reason>"}\n'
            "}"
        )

        user_content = (
            f"Context Category: {request.category or candidate.category or 'General'}\n"
            f"Hateful Comment: {request.query_text}\n"
            f"Candidate Counter-Speech (Generated by {candidate.model_id}):\n"
            f'"""{candidate.text}"""\n\n'
            f"Evaluate the candidate strictly against the rubrics."
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
                "maxOutputTokens": 400
            }
        }

        url = f"{self._base_url}/models/{self._model_id}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates_parts = data.get("candidates", [])
                raw_json_str = ""
                if candidates_parts:
                    parts = candidates_parts[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_json_str = parts[0].get("text", "").strip()

                if not raw_json_str:
                    return CandidateEvaluation(
                        candidate_id=candidate.candidate_id,
                        model_id=candidate.model_id,
                        evaluator_id=self.evaluator_id,
                        evaluator_type=self.evaluator_type,
                        schema_version="1.0",
                        scores={},
                        reasons={"error": "Empty response from Gemini judge."},
                        is_mock=False,
                        is_eligible=False,
                        ineligibility_reasons=["Empty response content from Gemini judge."],
                        aggregate_score=None,
                        evaluation_status="MALFORMED_SCORES",
                        timestamp=now_iso,
                        metadata={}
                    )

                # Parse JSON
                try:
                    parsed = json.loads(raw_json_str)
                except Exception:
                    match = re.search(r"\{.*\}", raw_json_str, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                    else:
                        raise ValueError(f"Could not parse judge JSON: {raw_json_str[:200]}")

                scores_raw = parsed.get("scores", {})
                reasons_raw = parsed.get("reasons", {})

                is_valid, err_msg, clean_scores = validate_scores(scores_raw)
                if not is_valid:
                    return CandidateEvaluation(
                        candidate_id=candidate.candidate_id,
                        model_id=candidate.model_id,
                        evaluator_id=self.evaluator_id,
                        evaluator_type=self.evaluator_type,
                        schema_version="1.0",
                        scores={},
                        reasons={"validation_error": err_msg or "Invalid scores dictionary"},
                        is_mock=False,
                        is_eligible=False,
                        ineligibility_reasons=[f"Invalid scores from judge: {err_msg}"],
                        aggregate_score=None,
                        evaluation_status="MALFORMED_SCORES",
                        timestamp=now_iso,
                        metadata={"raw_output": raw_json_str[:500]}
                    )

                clean_reasons = {
                    "relevance": str(reasons_raw.get("relevance", "Evaluated relevance.")),
                    "persuasiveness": str(reasons_raw.get("persuasiveness", "Evaluated persuasiveness.")),
                    "de_escalation": str(reasons_raw.get("de_escalation", "Evaluated de-escalation.")),
                    "safety": str(reasons_raw.get("safety", "Evaluated safety."))
                }

                return CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    model_id=candidate.model_id,
                    evaluator_id=self.evaluator_id,
                    evaluator_type=self.evaluator_type,
                    schema_version="1.0",
                    scores=clean_scores,
                    reasons=clean_reasons,
                    is_mock=False,
                    is_eligible=True,
                    ineligibility_reasons=[],
                    aggregate_score=None,
                    evaluation_status="SUCCESS",
                    timestamp=now_iso,
                    metadata={"model_id": self._model_id}
                )

        except urllib.error.HTTPError as he:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                model_id=candidate.model_id,
                evaluator_id=self.evaluator_id,
                evaluator_type=self.evaluator_type,
                schema_version="1.0",
                scores={},
                reasons={"http_error": f"HTTP {he.code}: {he.reason}"},
                is_mock=False,
                is_eligible=False,
                ineligibility_reasons=[f"Judge HTTP {he.code}: {he.reason}"],
                aggregate_score=None,
                evaluation_status="EVALUATION_ERROR",
                timestamp=now_iso,
                metadata={"status_code": he.code}
            )
        except Exception as exc:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                model_id=candidate.model_id,
                evaluator_id=self.evaluator_id,
                evaluator_type=self.evaluator_type,
                schema_version="1.0",
                scores={},
                reasons={"error": str(exc)},
                is_mock=False,
                is_eligible=False,
                ineligibility_reasons=[f"Judge error: {type(exc).__name__}: {str(exc)}"],
                aggregate_score=None,
                evaluation_status="EVALUATION_ERROR",
                timestamp=now_iso,
                metadata={"error_type": type(exc).__name__}
            )


# Backwards compatibility alias
LLMJudgeEvaluator = GeminiCandidateEvaluator
