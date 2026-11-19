"""
CSHSM Milestone 9: Five-Model Concurrent Generation Fan-Out.

Flow:
  Low retrieval similarity (M8 RoutingDecision)
      ↓
  Generation Fan-Out (Concurrent calls to up to 5 model slots)
      ↓
  Candidate Collection (Isolated exceptions, per-adapter timeouts, latency tracking)
      ↓
  Basic Structural Validation (Empty, malformed, duplicate, length constraints)
      ↓
  Safety Filtering (Conservative deterministic rules via src.kb.filters.safety_filter)
      ↓
  M10 Evaluation & Ranking (Explicitly deferred to Milestone 10)
      ↓
  Safe Fallback (Hardcoded safe response if all candidates fail or execution fails)
"""

import os
import time
import uuid
import hashlib
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.routing.router import RoutingDecision
from src.kb.filters import safety_filter
from src.generation.adapters import (
    BaseModelAdapter,
    AdapterOutput,
    MockModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleGeminiAdapter,
    MistralAdapter,
    CohereAdapter,
    CloudflareWorkersAIAdapter,
    GroqAdapter
)


# =====================================================================
# Request & Candidate Data Models
# =====================================================================

@dataclass
class GenerationRequest:
    """
    Standard request payload consumed by GenerationFanOut.
    Can be constructed directly or ingested from an M8 RoutingDecision.
    """
    query_text: str
    category: Optional[str] = None
    reference_context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_routing_decision(
        cls,
        decision: RoutingDecision,
        category: Optional[str] = None
    ) -> "GenerationRequest":
        """
        Ingests an M8 RoutingDecision fallback payload into a GenerationRequest.
        Fails safely if decision is malformed or not a fallback decision.
        """
        query_text = decision.query or ""
        ref_context = None

        if decision.fallback_payload:
            if not query_text and "query" in decision.fallback_payload:
                query_text = str(decision.fallback_payload["query"])

            # Extract reference counter-speech from top candidate if available
            top_cand = decision.fallback_payload.get("top_candidate")
            if isinstance(top_cand, dict):
                ref_context = top_cand.get("counter_speech")
                if not category:
                    category = top_cand.get("category")

        meta = {
            "source_action": decision.action,
            "fallback_reason": decision.fallback_reason,
            "threshold_applied": decision.threshold,
            "similarity_score": decision.similarity_score,
            "routing_provenance": decision.provenance
        }
        return cls(
            query_text=query_text,
            category=category,
            reference_context=ref_context,
            metadata=meta
        )


@dataclass
class GeneratedCandidate:
    """
    Normalized candidate schema for generated counter-speech responses.

    Safety & Review Guarantees:
      - quality_status is ALWAYS 'generated_candidate' (NEVER 'approved').
      - review_status is ALWAYS 'unreviewed' (NEVER 'approved').
      - human_audited is ALWAYS False.
      - Explicit unapproved disclaimer attached to every instance.
    """
    candidate_id: str
    text: str
    model_id: str
    provider: str
    is_mock: bool
    query_text: str
    category: Optional[str]
    latency_ms: float
    quality_status: str  # Always "generated_candidate"
    review_status: str   # Always "unreviewed"
    human_audited: bool  # Always False
    structural_validation: Dict[str, Any]
    safety_validation: Dict[str, Any]
    created_at: str
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterExecutionRecord:
    """
    Diagnostic telemetry tracking per-adapter execution.
    """
    slot_index: int
    model_id: str
    provider: str
    is_mock: bool
    status: str  # "SUCCESS" | "TIMEOUT" | "ADAPTER_ERROR" | "UNCONFIGURED_CREDENTIALS" | "EMPTY_OUTPUT"
    latency_ms: float
    error_category: Optional[str]
    error_message: Optional[str]
    candidate_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FanOutResult:
    """
    Complete outcome container of a multi-model generation fan-out execution.
    """
    request: GenerationRequest
    total_slots: int
    enabled_slots: int
    successful_slots: int
    failed_slots: int
    total_latency_ms: float
    execution_records: List[AdapterExecutionRecord]
    raw_candidates: List[GeneratedCandidate]
    valid_candidates: List[GeneratedCandidate]
    safe_candidates: List[GeneratedCandidate]
    rejected_candidates: List[GeneratedCandidate]
    status: str  # "SUCCESS" | "PARTIAL_SUCCESS" | "TOTAL_FAILURE" | "ALL_FILTERED"
    safe_fallback_triggered: bool
    fallback_response: Optional[str]
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["execution_records"] = [r.to_dict() for r in self.execution_records]
        d["raw_candidates"] = [c.to_dict() for c in self.raw_candidates]
        d["valid_candidates"] = [c.to_dict() for c in self.valid_candidates]
        d["safe_candidates"] = [c.to_dict() for c in self.safe_candidates]
        d["rejected_candidates"] = [c.to_dict() for c in self.rejected_candidates]
        return d


# =====================================================================
# Generation Fan-Out Engine
# =====================================================================

class GenerationFanOut:
    """
    Milestone 9 Concurrent Multi-Model Generation Fan-Out Engine.

    Supports up to 5 independently configured adapter slots.
    Performs concurrent execution, isolated error handling, structural validation,
    and conservative safety filtering before handing off to M10 evaluation.
    """

    DEFAULT_FALLBACK_RESPONSE = "I cannot respond to this respectfully."
    DISCLAIMER_UNAPPROVED = (
        "UNREVIEWED GENERATED CANDIDATE — Synthetic output pending M10 evaluation "
        "and final safety gating. Unapproved for production delivery."
    )

    def __init__(
        self,
        adapters: List[BaseModelAdapter],
        min_chars: int = 10,
        max_chars: int = 1000,
        overall_timeout: float = 70.0,
        fallback_message: str = DEFAULT_FALLBACK_RESPONSE
    ):
        if not adapters:
            raise ValueError("GenerationFanOut requires at least 1 configured model adapter.")
        if len(adapters) > 5:
            raise ValueError(f"GenerationFanOut supports at most 5 adapter slots, got {len(adapters)}.")

        self.adapters = list(adapters)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overall_timeout = overall_timeout
        self.fallback_message = fallback_message

    def execute(
        self,
        request_or_decision: Union[GenerationRequest, RoutingDecision],
        category_override: Optional[str] = None
    ) -> FanOutResult:
        """
        Executes concurrent fan-out across all configured adapter slots.
        Accepts either a GenerationRequest or an M8 RoutingDecision.
        """
        start_overall = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Ingest input safely
        if isinstance(request_or_decision, RoutingDecision):
            request = GenerationRequest.from_routing_decision(
                request_or_decision,
                category=category_override
            )
        elif isinstance(request_or_decision, GenerationRequest):
            request = request_or_decision
            if category_override:
                request.category = category_override
        else:
            raise TypeError(
                f"Expected GenerationRequest or RoutingDecision, got {type(request_or_decision).__name__}"
            )

        query_text = request.query_text.strip() if request.query_text else ""
        if not query_text:
            # Empty input triggers immediate total failure without invoking models
            rec = AdapterExecutionRecord(
                slot_index=0,
                model_id="none",
                provider="none",
                is_mock=True,
                status="EMPTY_INPUT",
                latency_ms=0.0,
                error_category="EMPTY_INPUT",
                error_message="Query text cannot be empty or whitespace.",
                candidate_id=None
            )
            return FanOutResult(
                request=request,
                total_slots=len(self.adapters),
                enabled_slots=len(self.adapters),
                successful_slots=0,
                failed_slots=len(self.adapters),
                total_latency_ms=0.0,
                execution_records=[rec],
                raw_candidates=[],
                valid_candidates=[],
                safe_candidates=[],
                rejected_candidates=[],
                status="TOTAL_FAILURE",
                safe_fallback_triggered=True,
                fallback_response=self.fallback_message,
                provenance={"error": "Empty query text", "timestamp": now_iso},
                disclaimer="SAFE FALLBACK — Empty query text supplied to generation fan-out."
            )

        # 1. Concurrent execution across all configured adapter slots
        execution_records: List[AdapterExecutionRecord] = []
        raw_outputs: List[tuple[int, BaseModelAdapter, AdapterOutput]] = []

        def _call_adapter(slot_idx: int, adapter: BaseModelAdapter) -> tuple[int, BaseModelAdapter, AdapterOutput]:
            try:
                out = adapter.generate(
                    prompt=query_text,
                    category=request.category,
                    timeout=adapter.timeout
                )
                return slot_idx, adapter, out
            except Exception as exc:
                # Absolute fail-safe: catch any unexpected exception from adapter implementation
                out = AdapterOutput(
                    model_id=adapter.model_id,
                    provider=adapter.provider,
                    is_mock=adapter.is_mock,
                    raw_text=None,
                    latency_ms=0.0,
                    status="ADAPTER_ERROR",
                    error_category="UNKNOWN_ERROR",
                    error_message=str(exc),
                    metadata={"error_type": type(exc).__name__}
                )
                return slot_idx, adapter, out

        executor = ThreadPoolExecutor(max_workers=len(self.adapters))
        futures = {
            executor.submit(_call_adapter, idx, adapter): (idx, adapter)
            for idx, adapter in enumerate(self.adapters)
        }
        completed_slots = set()

        try:
            for future in as_completed(futures.keys(), timeout=self.overall_timeout):
                idx, adapter = futures[future]
                try:
                    slot_idx, adapt, out = future.result()
                    raw_outputs.append((slot_idx, adapt, out))
                    completed_slots.add(slot_idx)
                except Exception as exc:
                    raw_outputs.append((idx, adapter, AdapterOutput(
                        model_id=adapter.model_id,
                        provider=adapter.provider,
                        is_mock=adapter.is_mock,
                        raw_text=None,
                        latency_ms=round((time.perf_counter() - start_overall) * 1000, 2),
                        status="ADAPTER_ERROR",
                        error_category="UNKNOWN_ERROR",
                        error_message=str(exc),
                        metadata={}
                    )))
                    completed_slots.add(idx)
        except TimeoutError:
            # Hard upper-bound reached: cancel unstarted futures and register timeout for incomplete slots
            for f in futures.keys():
                f.cancel()
            elapsed_ms = round((time.perf_counter() - start_overall) * 1000, 2)
            for f, (idx, adapter) in futures.items():
                if idx not in completed_slots:
                    raw_outputs.append((idx, adapter, AdapterOutput(
                        model_id=adapter.model_id,
                        provider=adapter.provider,
                        is_mock=adapter.is_mock,
                        raw_text=None,
                        latency_ms=elapsed_ms,
                        status="TIMEOUT",
                        error_category="TIMEOUT",
                        error_message=f"Fan-out overall timeout ({self.overall_timeout}s) exceeded.",
                        metadata={"overall_timeout_triggered": True}
                    )))
        finally:
            # Guarantee non-blocking shutdown: cancel pending work and do not hang waiting on unresponsive threads
            executor.shutdown(wait=False, cancel_futures=True)

        # Sort collected outputs by slot_index for determinism
        raw_outputs.sort(key=lambda item: item[0])

        # 2. Normalize outputs into Candidate Schema and track execution records
        raw_candidates: List[GeneratedCandidate] = []
        seen_texts = set()

        for slot_idx, adapter, out in raw_outputs:
            cand_id = None
            if out.status == "SUCCESS" and out.raw_text:
                cand_id = self._generate_stable_candidate_id(query_text, adapter.model_id, slot_idx)

            rec = AdapterExecutionRecord(
                slot_index=slot_idx,
                model_id=adapter.model_id,
                provider=adapter.provider,
                is_mock=adapter.is_mock,
                status=out.status,
                latency_ms=out.latency_ms,
                error_category=out.error_category,
                error_message=out.error_message,
                candidate_id=cand_id
            )
            execution_records.append(rec)

            if out.status == "SUCCESS" and out.raw_text:
                cand = self._build_candidate(
                    cand_id=cand_id,
                    raw_text=out.raw_text,
                    adapter=adapter,
                    query_text=query_text,
                    category=request.category,
                    latency_ms=out.latency_ms,
                    slot_idx=slot_idx,
                    seen_texts=seen_texts,
                    now_iso=now_iso
                )
                raw_candidates.append(cand)

        total_latency_ms = round((time.perf_counter() - start_overall) * 1000, 2)

        # 3. Separate Valid and Safe Candidates
        valid_candidates = [c for c in raw_candidates if c.structural_validation["is_valid"]]
        safe_candidates = [c for c in valid_candidates if c.safety_validation["is_safe"]]
        rejected_candidates = [c for c in raw_candidates if not c.structural_validation["is_valid"] or not c.safety_validation["is_safe"]]

        # 4. Determine Fan-Out Outcome Status
        enabled_count = len(self.adapters)
        successful_count = len([r for r in execution_records if r.status == "SUCCESS"])
        failed_count = enabled_count - successful_count

        if len(safe_candidates) == enabled_count:
            status = "SUCCESS"
        elif len(safe_candidates) > 0:
            status = "PARTIAL_SUCCESS"
        elif successful_count == 0:
            status = "TOTAL_FAILURE"
        else:
            status = "ALL_FILTERED"

        # 5. Safe Fallback Trigger
        safe_fallback_triggered = (len(safe_candidates) == 0)
        fallback_resp = self.fallback_message if safe_fallback_triggered else None

        provenance = {
            "executed_at": now_iso,
            "total_slots": enabled_count,
            "adapters": [
                {"slot": idx, "model_id": a.model_id, "provider": a.provider, "is_mock": a.is_mock}
                for idx, a in enumerate(self.adapters)
            ],
            "total_latency_ms": total_latency_ms,
            "status": status,
            "safe_fallback_triggered": safe_fallback_triggered
        }

        return FanOutResult(
            request=request,
            total_slots=enabled_count,
            enabled_slots=enabled_count,
            successful_slots=successful_count,
            failed_slots=failed_count,
            total_latency_ms=total_latency_ms,
            execution_records=execution_records,
            raw_candidates=raw_candidates,
            valid_candidates=valid_candidates,
            safe_candidates=safe_candidates,
            rejected_candidates=rejected_candidates,
            status=status,
            safe_fallback_triggered=safe_fallback_triggered,
            fallback_response=fallback_resp,
            provenance=provenance,
            disclaimer=self.DISCLAIMER_UNAPPROVED
        )

    def _generate_stable_candidate_id(self, query_text: str, model_id: str, slot_idx: int) -> str:
        """Generates a stable deterministic candidate identifier."""
        seed = f"{query_text}|{model_id}|{slot_idx}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        return f"GEN-{digest}"

    def _build_candidate(
        self,
        cand_id: str,
        raw_text: str,
        adapter: BaseModelAdapter,
        query_text: str,
        category: Optional[str],
        latency_ms: float,
        slot_idx: int,
        seen_texts: set,
        now_iso: str
    ) -> GeneratedCandidate:
        """Constructs a normalized GeneratedCandidate with structural validation and safety filtering."""
        cleaned_text = raw_text.strip()
        length_chars = len(cleaned_text)
        approx_tokens = len(cleaned_text.split())

        # Basic Structural Validation
        is_empty = (length_chars == 0)
        is_too_short = (length_chars < self.min_chars)
        is_too_long = (length_chars > self.max_chars)

        # Duplicate detection (normalized lower case)
        norm_key = re.sub(r"\s+", " ", cleaned_text.lower())
        is_dup = norm_key in seen_texts
        seen_texts.add(norm_key)

        is_structurally_valid = True
        val_reason = None

        if is_empty:
            is_structurally_valid = False
            val_reason = "EMPTY_TEXT"
        elif is_too_short:
            is_structurally_valid = False
            val_reason = "INSUFFICIENT_LENGTH"
        elif is_too_long:
            is_structurally_valid = False
            val_reason = "EXCESSIVE_LENGTH"
        elif is_dup:
            is_structurally_valid = False
            val_reason = "DUPLICATE_CANDIDATE"

        structural_val = {
            "is_valid": is_structurally_valid,
            "length_chars": length_chars,
            "length_tokens_approx": approx_tokens,
            "is_empty": is_empty,
            "is_excessively_long": is_too_long,
            "is_duplicate": is_dup,
            "validation_reason": val_reason
        }

        # Safety Filtering via established src.kb.filters.safety_filter
        is_safe, rejection_reason = safety_filter(cleaned_text)
        safety_val = {
            "is_safe": is_safe,
            "rejection_reason": rejection_reason if not is_safe else ""
        }

        provenance = {
            "slot_index": slot_idx,
            "model_id": adapter.model_id,
            "provider": adapter.provider,
            "is_mock": adapter.is_mock,
            "latency_ms": latency_ms,
            "created_at": now_iso
        }

        return GeneratedCandidate(
            candidate_id=cand_id,
            text=cleaned_text,
            model_id=adapter.model_id,
            provider=adapter.provider,
            is_mock=adapter.is_mock,
            query_text=query_text,
            category=category,
            latency_ms=latency_ms,
            quality_status="generated_candidate",
            review_status="unreviewed",
            human_audited=False,
            structural_validation=structural_val,
            safety_validation=safety_val,
            created_at=now_iso,
            provenance=provenance,
            disclaimer=self.DISCLAIMER_UNAPPROVED
        )


# =====================================================================
# Provider Configuration Helpers
# =====================================================================

def is_cloudflare_configured() -> bool:
    """
    Returns True if Cloudflare Workers AI credentials (account ID and API token)
    are present and non-empty in the environment.
    """
    return bool(
        os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        and os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    )


def is_groq_configured() -> bool:
    """
    Returns True if Groq API credentials (API key) are present and non-empty
    in the environment.
    """
    return bool(os.getenv("GROQ_API_KEY", "").strip())


# =====================================================================
# Factory Helper: 5-Model Fan-Out Builder
# =====================================================================

def create_default_fan_out(
    use_mock: bool = True,
    mock_responses: Optional[List[str]] = None,
    use_live_gemini: bool = False,
    use_legacy_models: bool = False,
    use_live_cloudflare: Optional[bool] = None,
    use_live_groq: Optional[bool] = None,
    allow_mock_fallback: bool = False,
    overall_timeout: float = 70.0
) -> GenerationFanOut:
    """
    Constructs a standard 5-model fan-out instance.

    Finalized Shortlist (Default):
      Slot 1: Gemma 4 26B IT (Cloudflare Workers AI: @cf/google/gemma-4-26b-a4b-it)
      Slot 2: GLM-4.7-Flash (Cloudflare Workers AI: @cf/zai-org/glm-4.7-flash)
      Slot 3: GPT-OSS-20B (Groq: openai/gpt-oss-20b)
      Slot 4: Qwen 3.8 27B (Groq: qwen/qwen3.8-27b)
      Slot 5: GPT-OSS-120B (Groq: openai/gpt-oss-120b)

    Modes:
      - All-Mock Shortlist (default: use_mock=True, use_live_gemini=False, use_legacy_models=False):
          5 deterministic mock adapters matching the finalized shortlist:
            Slot 1: mock-gemma4-26b-it
            Slot 2: mock-glm-4.7-flash
            Slot 3: mock-gpt-oss-20b
            Slot 4: mock-qwen-3.8-27b
            Slot 5: mock-gpt-oss-120b

      - Full Live Shortlist (use_mock=False):
          5 genuine Cloudflare & Groq provider adapters.
          If credentials are missing and allow_mock_fallback is False, adapters fail closed
          with UNCONFIGURED_CREDENTIALS without masking as mock.

      - Partial-Live Execution (use_live_cloudflare / use_live_groq or allow_mock_fallback=True):
          Slots with verified credentials use genuine live adapters (is_mock=False).
          Slots with missing credentials fall back to mock adapters (is_mock=True) ONLY
          when allow_mock_fallback=True, preserving honest provenance.

      - Legacy Hybrid Gemini-Live (use_live_gemini=True):
          Retained for backwards-compatibility: Slot 3 uses real GoogleGeminiAdapter.

      - Legacy Models (use_legacy_models=True):
          Retained for backwards-compatibility: OpenAI, Anthropic, Gemini, Mistral, Cohere.
    """
    default_responses = [
        "Diversity of perspectives strengthens society. Respectful discussion is always more productive than hatred.",
        "Dehumanizing entire groups harms everyone. Let us focus on individual actions rather than broad prejudice.",
        "Constructive disagreement is natural, but hostility and slurs only deepen misunderstandings.",
        "Everyone deserves dignity and safety regardless of their background or identity.",
        "Stereotyping creates unnecessary divisions. Empathy and factual understanding build healthier communities."
    ]
    responses = mock_responses or default_responses

    if use_live_gemini:
        adapters: List[BaseModelAdapter] = [
            MockModelAdapter(model_id="mock-openai-gpt4o-mini", provider="mock", canned_response=responses[0 % len(responses)]),
            MockModelAdapter(model_id="mock-anthropic-claude3-haiku", provider="mock", canned_response=responses[1 % len(responses)]),
            GoogleGeminiAdapter(model_id="gemini-3.8-flash"),
            MockModelAdapter(model_id="mock-mistral-small", provider="mock", canned_response=responses[3 % len(responses)]),
            MockModelAdapter(model_id="mock-cohere-command-r", provider="mock", canned_response=responses[4 % len(responses)])
        ]
    elif use_legacy_models:
        if use_mock:
            adapters = [
                MockModelAdapter(model_id="mock-openai-gpt4o-mini", provider="mock", canned_response=responses[0 % len(responses)]),
                MockModelAdapter(model_id="mock-anthropic-claude3-haiku", provider="mock", canned_response=responses[1 % len(responses)]),
                MockModelAdapter(model_id="mock-google-gemini15-flash", provider="mock", canned_response=responses[2 % len(responses)]),
                MockModelAdapter(model_id="mock-mistral-small", provider="mock", canned_response=responses[3 % len(responses)]),
                MockModelAdapter(model_id="mock-cohere-command-r", provider="mock", canned_response=responses[4 % len(responses)])
            ]
        else:
            adapters = [
                OpenAIAdapter(model_id="gpt-4o-mini"),
                AnthropicAdapter(model_id="claude-3-haiku-20240307"),
                GoogleGeminiAdapter(model_id="gemini-3.8-flash"),
                MistralAdapter(model_id="mistral-small-latest"),
                CohereAdapter(model_id="command-r")
            ]
    else:
        # Finalized 5-Generator Roster with Per-Provider Live / Mock Granularity
        want_live_cf = (not use_mock) if use_live_cloudflare is None else use_live_cloudflare
        want_live_groq = (not use_mock) if use_live_groq is None else use_live_groq

        cf_ready = is_cloudflare_configured()
        groq_ready = is_groq_configured()

        # Resolve Cloudflare Slots 1 and 2
        if want_live_cf:
            if cf_ready or not allow_mock_fallback:
                slot1 = CloudflareWorkersAIAdapter(model_id="@cf/google/gemma-4-26b-a4b-it")
                slot2 = CloudflareWorkersAIAdapter(model_id="@cf/zai-org/glm-4.7-flash")
            else:
                slot1 = MockModelAdapter(model_id="mock-gemma4-26b-it", provider="mock", canned_response=responses[0 % len(responses)])
                slot2 = MockModelAdapter(model_id="mock-glm-4.7-flash", provider="mock", canned_response=responses[1 % len(responses)])
        else:
            slot1 = MockModelAdapter(model_id="mock-gemma4-26b-it", provider="mock", canned_response=responses[0 % len(responses)])
            slot2 = MockModelAdapter(model_id="mock-glm-4.7-flash", provider="mock", canned_response=responses[1 % len(responses)])

        # Resolve Groq Slots 3, 4, and 5
        if want_live_groq:
            if groq_ready or not allow_mock_fallback:
                slot3 = GroqAdapter(model_id="openai/gpt-oss-20b")
                slot4 = GroqAdapter(model_id="qwen/qwen3.8-27b")
                slot5 = GroqAdapter(model_id="openai/gpt-oss-120b")
            else:
                slot3 = MockModelAdapter(model_id="mock-gpt-oss-20b", provider="mock", canned_response=responses[2 % len(responses)])
                slot4 = MockModelAdapter(model_id="mock-qwen-3.8-27b", provider="mock", canned_response=responses[3 % len(responses)])
                slot5 = MockModelAdapter(model_id="mock-gpt-oss-120b", provider="mock", canned_response=responses[4 % len(responses)])
        else:
            slot3 = MockModelAdapter(model_id="mock-gpt-oss-20b", provider="mock", canned_response=responses[2 % len(responses)])
            slot4 = MockModelAdapter(model_id="mock-qwen-3.8-27b", provider="mock", canned_response=responses[3 % len(responses)])
            slot5 = MockModelAdapter(model_id="mock-gpt-oss-120b", provider="mock", canned_response=responses[4 % len(responses)])

        adapters = [slot1, slot2, slot3, slot4, slot5]

    return GenerationFanOut(adapters=adapters, overall_timeout=overall_timeout)
