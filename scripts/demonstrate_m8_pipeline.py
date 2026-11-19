#!/usr/bin/env python3
"""
CSHSM Milestone 8: Finalized End-to-End Pipeline Demonstration Harness.

Integrates:
  1. Multitask XLM-R Classifier (models/xlmr_best.pt on CUDA/CPU, or MockClassifier)
  2. BGE-M3 Dense Embedder (BAAI/bge-m3, 1024-dim, local_files_only=True, or MockEmbedder)
  3. Authoritative KB V2 Retrieval (FAISS IndexFlatIP on data/counter_speech_kb/embeddings_v2, 300 records)
  4. FallbackRouter (calibrated similarity threshold tau = 0.80)
  5. 5-Model Concurrent Generation Fan-Out:
       - Slot 1: Gemma 4 26B IT (Cloudflare Workers AI: @cf/google/gemma-4-26b-a4b-it)
       - Slot 2: GLM-4.7-Flash (Cloudflare Workers AI: @cf/zai-org/glm-4.7-flash)
       - Slot 3: GPT-OSS-20B (Groq: openai/gpt-oss-20b)
       - Slot 4: Qwen 3.8 27B (Groq: qwen/qwen3.8-27b)
       - Slot 5: GPT-OSS-120B (Groq: openai/gpt-oss-120b)
       With 70.0s hard ceiling and non-blocking executor cleanup.
  6. Dedicated LLM Judge Evaluator:
       - GeminiCandidateEvaluator (gemini-flash-lite-latest, GOOGLE_API_KEY)
       - Strict self-evaluation prohibition (Gemini never evaluates itself)
       - Strict fail-closed error handling
  7. Multi-Criteria CandidateRanker (mandatory hard gates: Safety>=4, Relevance>=3, De-escalation>=3, Persuasiveness>=1)
  8. Defense-in-Depth FinalSafetyGate (rejection triggers safe fallback)
  9. PipelineOrchestrator (12-stage workflow coordination)

Safety Invariants:
  - Defaults to completely offline mock mode (use_mock=True).
  - Explicit --live CLI flag required for live provider execution.
  - Pre-flight checks verify all credentials and artifacts before any live call.
  - Strict anti-fallback guard: Refuses to fall back silently to the 24-entry demo KB.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv

# Ensure project root is on sys.path and load environment variables from .env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from src.data.cleaner import TextCleaner
from src.pipeline.classifier import BaseClassifier, XLMROracleClassifier, MockClassifier, ClassifierOutput
from src.retrieval.embedder import BaseEmbedder, BGE_M3_Embedder, MockEmbedder, EmbeddingPipeline
from src.retrieval.retriever import CSHSMRetriever
from src.routing.router import FallbackRouter
from src.generation.adapters import BaseModelAdapter, MockModelAdapter
from src.generation.fan_out import (
    GenerationFanOut,
    create_default_fan_out,
    is_cloudflare_configured,
    is_groq_configured
)
from src.evaluation.evaluator import BaseCandidateEvaluator, GeminiCandidateEvaluator, MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker
from src.safety.final_gate import FinalSafetyGate
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineResult, create_mock_retriever


# =====================================================================
# KB V2 Verification Guard
# =====================================================================

def _verify_kb_v2_artifacts(kb_output_dir: str) -> None:
    """
    Verifies that the required Knowledge Base V2 artifacts exist and are committed.
    Fails closed immediately rather than allowing silent fallback to demo KB.
    """
    if not os.path.isdir(kb_output_dir):
        raise FileNotFoundError(
            f"CSHSM M8 requires KB V2 embeddings directory at '{kb_output_dir}'. "
            "Directory does not exist. Silent fallback to demo KB is strictly prohibited."
        )

    pointer_path = os.path.join(kb_output_dir, "LATEST.json")
    if not os.path.isfile(pointer_path):
        raise FileNotFoundError(
            f"CSHSM M8 requires authoritative KB V2 pointer at '{pointer_path}'. "
            "File does not exist. Refusing to fall back to demo KB."
        )

    # Use EmbeddingPipeline.resolve_latest_run to verify pointer, run dir, COMMITTED marker, and SHA-256 hashes
    resolved = EmbeddingPipeline.resolve_latest_run(output_dir=kb_output_dir, allow_mock=False)
    vector_count = resolved.get("vector_count", 0)
    if vector_count != 300:
        raise ValueError(
            f"CSHSM M8 requires exactly 300 committed KB V2 vectors, but resolved run has {vector_count}."
        )


# =====================================================================
# M8 Pipeline Factory
# =====================================================================

def build_m8_pipeline(
    use_mock: bool = True,
    use_mock_judge: Optional[bool] = None,
    threshold: float = 0.80,
    kb_output_dir: str = "data/counter_speech_kb/embeddings_v2",
    device: Optional[str] = None,
    gemini_api_key_env_var: str = "GOOGLE_API_KEY",
    classifier: Optional[BaseClassifier] = None,
    retriever: Optional[CSHSMRetriever] = None,
    fan_out: Optional[GenerationFanOut] = None,
    evaluator: Optional[BaseCandidateEvaluator] = None,
    cleaner: Optional[TextCleaner] = None,
    fallback_message: str = PipelineOrchestrator.DEFAULT_FALLBACK_MESSAGE,
    use_real_kb: bool = True,
    use_real_classifier: bool = False,
    use_live_cloudflare: Optional[bool] = None,
    use_live_groq: Optional[bool] = None,
    allow_mock_fallback: bool = False,
    allow_mock: Optional[bool] = None
) -> PipelineOrchestrator:
    """
    Constructs the authoritative Milestone 8 CSHSM Pipeline Orchestrator.

    Args:
      use_mock: If True, operates completely offline with deterministic mock components.
      use_mock_judge: If provided, explicitly toggles mock status for evaluator.
      threshold: Calibrated semantic similarity threshold tau (default: 0.80).
      kb_output_dir: Path to committed KB V2 embeddings directory (default: data/counter_speech_kb/embeddings_v2).
      device: Hardware device for XLM-R and BGE-M3 ('cuda', 'cpu', or auto-detect).
      gemini_api_key_env_var: Environment variable name for Gemini API key (default: GOOGLE_API_KEY).
      classifier: Optional pre-constructed classifier override.
      retriever: Optional pre-constructed retriever override.
      fan_out: Optional pre-constructed fan-out override.
      evaluator: Optional pre-constructed evaluator override.
      cleaner: Optional pre-constructed cleaner override.
      fallback_message: Standard fallback message on rejection or failure.
      use_real_kb: If True (default), connects to the real committed 300-record KB V2 with local BGE-M3 and FAISS.
      use_real_classifier: If True (or CSHSM_M8_USE_REAL_CLASSIFIER=1), loads real local XLM-R classifier.
      use_live_cloudflare: Explicitly enable live Cloudflare Workers AI generation adapters.
      use_live_groq: Explicitly enable live Groq generation adapters.
      allow_mock_fallback: If True, allows unconfigured live slots to fall back to mock adapters.
      allow_mock: Override for PipelineOrchestrator allow_mock guard.

    Returns:
      Fully configured PipelineOrchestrator instance.
    """
    cleaner = cleaner or TextCleaner()
    judge_is_mock = use_mock if use_mock_judge is None else use_mock_judge

    # Environment toggle: CSHSM_M8_USE_MOCK_KB=1 forces 2-record mock retriever if explicitly requested
    use_real_kb_env = os.getenv("CSHSM_M8_USE_MOCK_KB", "").strip().lower() not in ("1", "true", "yes")
    effective_use_real_kb = use_real_kb and use_real_kb_env

    # Environment toggle: CSHSM_M8_USE_REAL_CLASSIFIER=1 enables real XLM-R classifier in offline mock mode
    use_real_classifier_env = os.getenv("CSHSM_M8_USE_REAL_CLASSIFIER", "").strip().lower() in ("1", "true", "yes")
    effective_use_real_classifier = use_real_classifier or use_real_classifier_env

    canned = {
        "Wishing everyone a peaceful and productive week ahead! Let's collaborate together.": ClassifierOutput(
            is_hate=False,
            category=None,
            confidence=0.99,
            model_id="mock-xlmr-multitask",
            is_mock=True,
            metadata={"reason": "preset_non_hate"}
        ),
        "Have a wonderful and blessed day with your family!": ClassifierOutput(
            is_hate=False,
            category=None,
            confidence=0.99,
            model_id="mock-xlmr-multitask",
            is_mock=True,
            metadata={"reason": "preset_non_hate"}
        ),
        "Have a wonderful, peaceful, and blessed day with your family!": ClassifierOutput(
            is_hate=False,
            category=None,
            confidence=0.99,
            model_id="mock-xlmr-multitask",
            is_mock=True,
            metadata={"reason": "preset_non_hate"}
        ),
        "Hudugir ge driving baralla, road alli bandre accident guarantee.": ClassifierOutput(
            is_hate=True,
            category="Gender",
            confidence=0.98,
            model_id="mock-xlmr-multitask",
            is_mock=True,
            metadata={"reason": "preset_gender_hate"}
        ),
        "Girls ge coding antha tough subject ella, boys ge matra sari hogtade.": ClassifierOutput(
            is_hate=True,
            category="Gender",
            confidence=0.98,
            model_id="mock-xlmr-multitask",
            is_mock=True,
            metadata={"reason": "preset_gender_hate"}
        )
    }

    if classifier is None:
        if effective_use_real_classifier or (not use_mock and not allow_mock_fallback):
            try:
                classifier = XLMROracleClassifier(
                    model_path="models/xlmr_best.pt",
                    device=device
                )
            except Exception as exc:
                if not use_mock and not allow_mock_fallback:
                    raise RuntimeError(
                        f"Failed to initialize real XLM-R classifier from 'models/xlmr_best.pt': {exc}"
                    ) from exc
                classifier = MockClassifier(
                    default_is_hate=True,
                    default_category="Religion",
                    default_confidence=0.95,
                    canned_predictions=canned
                )
        else:
            classifier = MockClassifier(
                default_is_hate=True,
                default_category="Religion",
                default_confidence=0.95,
                canned_predictions=canned
            )

    if retriever is None:
        if effective_use_real_kb:
            # Load committed 300-record KB V2 using local BGE-M3 and FAISS
            _verify_kb_v2_artifacts(kb_output_dir)
            import torch
            chosen_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            embedder = BGE_M3_Embedder(
                model_name="BAAI/bge-m3",
                device=chosen_device,
                local_files_only=True
            )
            retriever = CSHSMRetriever.from_committed_run(
                output_dir=kb_output_dir,
                allow_mock=False,
                embedder=embedder,
                force_backend="faiss"
            )
            if retriever.index.ntotal != 300:
                raise RuntimeError(
                    f"KB V2 integrity violation: expected 300 entries in FAISS index, found {retriever.index.ntotal}."
                )
        else:
            # Deterministic mock retriever containing sample KB entries matching V2 schema
            mock_entries = [
                {
                    "kb_id": "KB-V2-00001",
                    "hate_text": "Hudugir ge driving baralla, road alli bandre accident guarantee.",
                    "counter_speech": "Driving skill gender mele depend aagalla. Proper practice, road awareness mattu traffic rules paalisidre yaaru aadru safe driver aagbahudu. Bengaluru traffic nalli saaviraaru mahileyaru two-wheeler, cab mattu BMTC bus galannu safely operate maadtiro real examples namma mundide.",
                    "category": "Gender",
                    "target": "women",
                    "source": "cshsm_v2",
                    "source_id": "V2-001",
                    "quality_status": "approved_candidate",
                    "review_status": "unreviewed",
                    "scope": "development_provisional",
                    "row_idx": 0
                },
                {
                    "kb_id": "KB-V2-00002",
                    "hate_text": "Sample religious hostility query",
                    "counter_speech": "Freedom of religion is a human right. Mutual respect strengthens us all.",
                    "category": "Religion",
                    "target": "religion",
                    "source": "cshsm_v2",
                    "source_id": "V2-002",
                    "quality_status": "approved_candidate",
                    "review_status": "unreviewed",
                    "scope": "development_provisional",
                    "row_idx": 1
                }
            ]
            retriever = create_mock_retriever(entries=mock_entries)

    if fan_out is None:
        fan_out = create_default_fan_out(
            use_mock=use_mock,
            use_live_cloudflare=use_live_cloudflare,
            use_live_groq=use_live_groq,
            allow_mock_fallback=allow_mock_fallback
        )

    effective_allow_mock = allow_mock if allow_mock is not None else (use_mock or allow_mock_fallback)

    # Router with calibrated similarity threshold tau = 0.80
    router = FallbackRouter(retriever=retriever, default_threshold=threshold)

    # Evaluator: dedicated Gemini Flash-Lite LLM Judge or deterministic mock
    if evaluator is None:
        if judge_is_mock:
            evaluator = MockCandidateEvaluator()
        else:
            resolved_key_var = gemini_api_key_env_var
            if not os.getenv(resolved_key_var) and os.getenv("GEMINI_API_KEY"):
                resolved_key_var = "GEMINI_API_KEY"

            evaluator = GeminiCandidateEvaluator(
                model_id="gemini-flash-lite-latest",
                api_key_env_var=resolved_key_var,
                timeout=30.0
            )

    ranker = CandidateRanker(evaluator=evaluator)
    final_gate = FinalSafetyGate(fallback_message=fallback_message)

    return PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=cleaner,
        allow_mock=effective_allow_mock,
        threshold=threshold,
        fallback_message=fallback_message
    )


# =====================================================================
# Operational Demonstration Scenarios (Scenarios 1 through 8)
# =====================================================================

def get_demonstration_scenarios() -> List[Dict[str, Any]]:
    """Returns the standardized 8 operational demonstration scenarios."""
    return [
        {
            "id": 1,
            "name": "Non-hate early exit",
            "input": "Wishing everyone a peaceful and productive week ahead! Let's collaborate together.",
            "description": "Benign positive statement; classifier classifies as NON-HATE and exits early without generating counter-speech."
        },
        {
            "id": 2,
            "name": "Successful KB V2 retrieval",
            "input": "Girls ge coding antha tough subject ella, boys ge matra sari hogtade.",
            "description": "Exact match to CSHSM-KN-00003; router detects similarity >= 0.80 and routes to retrieved response without generator calls."
        },
        {
            "id": 3,
            "name": "Five-model generation fan-out fallback",
            "input": "Every single member of that community is a criminal and should be deported immediately.",
            "description": "Novel hate speech; similarity < 0.80 triggers concurrent fan-out across all 5 models."
        },
        {
            "id": 4,
            "name": "Partial provider failure resiliency",
            "input": "That entire political party is corrupt and destroying our state.",
            "description": "Simulates 2 provider timeouts/failures; remaining 3 valid candidates are collected, evaluated, and ranked."
        },
        {
            "id": 5,
            "name": "Gemini evaluation failure fail-closed handling",
            "input": "People belonging to that religious sect have no moral integrity.",
            "description": "Judge experiences an unexpected network/schema error; candidate is marked ineligible and safe fallback is triggered."
        },
        {
            "id": 6,
            "name": "All candidates rejected by hard gates",
            "input": "Those people from the neighbouring region do not belong in our city.",
            "description": "All generated candidates fail mandatory hard gates (Safety < 4 or Relevance < 3); pipeline safely triggers fallback."
        },
        {
            "id": 7,
            "name": "Final safety gate defense-in-depth rejection",
            "input": "All members of that minority are evil subhumans who must be expelled.",
            "description": "Candidate passes ranker but fails stricter independent FinalSafetyGate threshold; gate intercepts and blocks delivery."
        },
        {
            "id": 8,
            "name": "Empty and whitespace input safe handling",
            "input": "   \t\n  ",
            "description": "Whitespace only input; cleaner catches empty string before classifier, returning safe fallback without error."
        }
    ]


def run_scenarios(
    orchestrator: PipelineOrchestrator,
    use_mock: bool = True,
    scenario_id_filter: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Executes the 8 operational demonstration scenarios and records detailed telemetry.
    """
    all_scenarios = get_demonstration_scenarios()
    if scenario_id_filter is None:
        scenarios_to_run = all_scenarios
    elif isinstance(scenario_id_filter, (list, tuple, set)):
        scenarios_to_run = [s for s in all_scenarios if s["id"] in scenario_id_filter]
    else:
        scenarios_to_run = [s for s in all_scenarios if s["id"] == scenario_id_filter]

    results = []

    print("\n" + "=" * 80)
    print(f"CSHSM M8 PIPELINE VERIFICATION ({'MOCK MODE' if use_mock else 'LIVE MODE'})")
    print(f"Total Scenarios to Execute: {len(scenarios_to_run)}")
    print("=" * 80)

    for sc in scenarios_to_run:
        sc_id = sc["id"]
        sc_name = sc["name"]
        inp = sc["input"]

        print(f"\n--- Scenario {sc_id}: {sc_name} ---")
        print(f"Input: \"{inp}\"")

        t0 = time.perf_counter()

        if sc_id == 4:
            # Scenario 4: Partial provider failure test
            # Construct a fan-out with 2 failing adapters and 3 succeeding adapters
            failing_adapters: List[BaseModelAdapter] = [
                MockModelAdapter(
                    model_id="mock-gemma4-26b-it",
                    provider="mock",
                    exception_to_raise=TimeoutError("Simulated Cloudflare timeout")
                ),
                MockModelAdapter(
                    model_id="mock-glm-4.7-flash",
                    provider="mock",
                    exception_to_raise=RuntimeError("Simulated provider 500 error")
                ),
                MockModelAdapter(
                    model_id="mock-gpt-oss-20b",
                    provider="mock",
                    canned_response="Constructive dialogue helps resolve political disagreements without broad demonization."
                ),
                MockModelAdapter(
                    model_id="mock-qwen-3.8-27b",
                    provider="mock",
                    canned_response="Individual accountability is essential rather than condemning an entire collective."
                ),
                MockModelAdapter(
                    model_id="mock-gpt-oss-120b",
                    provider="mock",
                    canned_response="Respectful civic participation leads to lasting progress and mutual understanding."
                )
            ]
            custom_fan_out = GenerationFanOut(adapters=failing_adapters, overall_timeout=70.0)
            mock_ranker = CandidateRanker(evaluator=MockCandidateEvaluator())
            custom_orch = PipelineOrchestrator(
                classifier=orchestrator.classifier,
                router=FallbackRouter(retriever=orchestrator.router.retriever, default_threshold=0.99),
                fan_out=custom_fan_out,
                ranker=mock_ranker,
                final_gate=orchestrator.final_gate,
                cleaner=orchestrator.cleaner,
                allow_mock=True,
                threshold=0.99
            )
            res = custom_orch.process(inp)

        elif sc_id == 5:
            # Scenario 5: Gemini evaluation failure fail-closed test
            broken_evaluator = MockCandidateEvaluator(
                exception_to_raise=RuntimeError("Simulated Gemini API 503 Service Unavailable")
            )
            broken_ranker = CandidateRanker(evaluator=broken_evaluator)
            custom_orch = PipelineOrchestrator(
                classifier=orchestrator.classifier,
                router=FallbackRouter(retriever=orchestrator.router.retriever, default_threshold=0.99),
                fan_out=orchestrator.fan_out,
                ranker=broken_ranker,
                final_gate=orchestrator.final_gate,
                cleaner=orchestrator.cleaner,
                allow_mock=True,
                threshold=0.99
            )
            res = custom_orch.process(inp)

        elif sc_id == 6:
            # Scenario 6: All candidates rejected by hard gates
            strict_evaluator = MockCandidateEvaluator(
                default_scores={"safety": 2, "relevance": 2, "de_escalation": 2, "persuasiveness": 2}
            )
            strict_ranker = CandidateRanker(evaluator=strict_evaluator)
            custom_orch = PipelineOrchestrator(
                classifier=orchestrator.classifier,
                router=FallbackRouter(retriever=orchestrator.router.retriever, default_threshold=0.99),
                fan_out=orchestrator.fan_out,
                ranker=strict_ranker,
                final_gate=orchestrator.final_gate,
                cleaner=orchestrator.cleaner,
                allow_mock=True,
                threshold=0.99
            )
            res = custom_orch.process(inp)

        elif sc_id == 7:
            # Scenario 7: Final safety gate defense-in-depth rejection
            # Candidate passes M10 ranking with score 4 (threshold=4), but independent FinalSafetyGate requires 5 -> gate rejects
            eval_score_4 = MockCandidateEvaluator(
                default_scores={"safety": 4, "relevance": 4, "de_escalation": 4, "persuasiveness": 4}
            )
            ranker_pass_4 = CandidateRanker(evaluator=eval_score_4)
            stricter_gate = FinalSafetyGate(min_safety_score=5)
            custom_orch = PipelineOrchestrator(
                classifier=orchestrator.classifier,
                router=FallbackRouter(retriever=orchestrator.router.retriever, default_threshold=0.99),
                fan_out=orchestrator.fan_out,
                ranker=ranker_pass_4,
                final_gate=stricter_gate,
                cleaner=orchestrator.cleaner,
                allow_mock=True,
                threshold=0.99
            )
            res = custom_orch.process(inp)

        else:
            # Scenarios 1, 2, 3, 8: Standard pipeline execution
            res = orchestrator.process(inp)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        print(f"Outcome: {res.outcome}")
        print(f"Is Hate: {res.is_hate} | Category: {res.category} | Confidence: {res.confidence:.4f}")
        print(f"Source Origin: {res.source_origin} | Candidate ID: {res.candidate_id} | Provider: {res.provider}")
        print(f"Safe Fallback: {res.safe_fallback_triggered} | Reason: {res.fallback_reason}")
        print(f"Delivered Response: \"{res.response_text or res.fallback_response}\"")
        print(f"Component Statuses: {res.component_statuses}")
        print(f"Latency: {latency_ms} ms")

        results.append({
            "scenario_id": sc_id,
            "name": sc_name,
            "description": sc["description"],
            "input_text": inp,
            "outcome": res.outcome,
            "is_hate": res.is_hate,
            "category": res.category,
            "confidence": res.confidence,
            "source_origin": res.source_origin,
            "candidate_id": res.candidate_id,
            "provider": res.provider,
            "model_id": res.model_id,
            "safe_fallback_triggered": res.safe_fallback_triggered,
            "fallback_reason": res.fallback_reason,
            "response_text": res.response_text,
            "fallback_response": res.fallback_response,
            "component_statuses": res.component_statuses,
            "provenance": res.provenance,
            "latency_ms": latency_ms
        })

    return results


# =====================================================================
# CLI Entry Point with Explicit Live Argument Guard
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CSHSM Milestone 8: Finalized End-to-End Pipeline Demonstration Harness"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Execute in full live mode with real models and external APIs. (Default: offline mock mode)"
    )
    parser.add_argument(
        "--partial-live",
        action="store_true",
        default=False,
        help="Execute in resilient partial-live mode: live providers for configured credentials, mock fallback for missing ones."
    )
    parser.add_argument(
        "--live-cloudflare",
        action="store_true",
        default=None,
        help="Explicitly enable live Cloudflare Workers AI generation adapters."
    )
    parser.add_argument(
        "--live-groq",
        action="store_true",
        default=None,
        help="Explicitly enable live Groq generation adapters."
    )
    parser.add_argument(
        "--mock-judge",
        action="store_true",
        default=False,
        help="Force mock evaluator judge even when running partial-live generation."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Execute in mock mode with zero external network or GPU calls. (Default: True)"
    )
    parser.add_argument(
        "--real-classifier",
        action="store_true",
        default=False,
        help="Use real local XLM-R classifier while keeping generation and evaluation in mock mode."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a specific scenario ID (1 to 8) or comma-separated list of IDs (e.g. '1,2,3')."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Evaluate an arbitrary input string through the pipeline."
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Path to save the JSON execution verification report."
    )
    parser.add_argument(
        "--kb-output-dir",
        type=str,
        default="data/counter_speech_kb/embeddings_v2",
        help="Directory containing committed KB V2 embeddings."
    )

    args = parser.parse_args()

    # Determine execution mode
    use_live = bool(args.live)
    use_partial_live = bool(args.partial_live)

    if use_live and use_partial_live:
        print("ERROR: Cannot specify both --live (strict all-live) and --partial-live simultaneously. Choose one.")
        sys.exit(1)

    has_cf = is_cloudflare_configured()
    has_groq = is_groq_configured()
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

    if use_live:
        print("[PRE-FLIGHT] Verifying requirements for FULL LIVE execution...")
        # 1. Model checkpoint
        if not os.path.isfile("models/xlmr_best.pt"):
            print("ERROR: Missing required XLM-R model weights at 'models/xlmr_best.pt'. Aborting.")
            sys.exit(1)

        # 2. KB V2 artifacts
        try:
            _verify_kb_v2_artifacts(args.kb_output_dir)
        except Exception as exc:
            print(f"ERROR: KB V2 artifact verification failed: {exc}. Aborting.")
            sys.exit(1)

        # 3. Environment credentials
        missing_creds = []
        if not has_cf:
            missing_creds.append("CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID")
        if not has_groq:
            missing_creds.append("GROQ_API_KEY")
        if not has_gemini:
            missing_creds.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")

        if missing_creds:
            print(f"ERROR: Missing live credentials in environment: {', '.join(missing_creds)}. Aborting.")
            sys.exit(1)

        print("[PRE-FLIGHT] All live prerequisites satisfied.")
        use_mock = False
        use_mock_judge = False
        allow_mock_fallback = False
        execution_mode = "live"

    elif use_partial_live:
        print("[PRE-FLIGHT] Configuring RESILIENT PARTIAL-LIVE execution...")
        cf_status = "LIVE (configured)" if has_cf else "MOCK (fallback: unconfigured)"
        groq_status = "LIVE (configured)" if has_groq else "MOCK (fallback: unconfigured)"
        gemini_status = "LIVE (configured)" if (has_gemini and not args.mock_judge) else "MOCK (fallback)"

        print(f"  - Cloudflare (Slots 1 & 2): {cf_status}")
        print(f"  - Groq (Slots 3, 4, 5):      {groq_status}")
        print(f"  - Evaluator Judge:           {gemini_status}")

        use_mock = False
        use_mock_judge = args.mock_judge or (not has_gemini)
        allow_mock_fallback = True
        execution_mode = "partial_live"

    else:
        # Full offline mock mode (default)
        use_mock = True
        use_mock_judge = True
        allow_mock_fallback = False
        execution_mode = "mock"

    # Build M8 pipeline
    orchestrator = build_m8_pipeline(
        use_mock=use_mock,
        use_mock_judge=use_mock_judge,
        kb_output_dir=args.kb_output_dir,
        use_real_classifier=bool(args.real_classifier),
        use_live_cloudflare=args.live_cloudflare,
        use_live_groq=args.live_groq,
        allow_mock_fallback=allow_mock_fallback
    )

    if args.input:
        print(f"\nProcessing single input text: \"{args.input}\"")
        res = orchestrator.process(args.input)
        print(f"Outcome: {res.outcome}")
        print(f"Response: {res.response_text or res.fallback_response}")
        return

    # Parse scenario filter if specified
    scenario_id_filter = None
    if args.scenario:
        try:
            if "," in args.scenario:
                scenario_id_filter = [int(s.strip()) for s in args.scenario.split(",") if s.strip()]
            else:
                scenario_id_filter = int(args.scenario.strip())
        except ValueError:
            print(f"ERROR: Invalid scenario argument '{args.scenario}'. Expected integer or comma-separated integers.")
            sys.exit(1)

    # Run operational scenarios
    scenario_results = run_scenarios(
        orchestrator=orchestrator,
        use_mock=use_mock,
        scenario_id_filter=scenario_id_filter
    )

    # Output JSON report if requested
    if args.json_report:
        report_data = {
            "milestone": "M8",
            "title": "CSHSM Milestone 8 Demonstration & Verification Report",
            "execution_mode": execution_mode,
            "provider_configuration": {
                "cloudflare_live": bool(args.live_cloudflare if args.live_cloudflare is not None else (has_cf and not use_mock)),
                "groq_live": bool(args.live_groq if args.live_groq is not None else (has_groq and not use_mock)),
                "gemini_live": bool(has_gemini and not use_mock_judge),
                "allow_mock_fallback": allow_mock_fallback
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios_executed": len(scenario_results),
            "results": scenario_results
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_report)), exist_ok=True)
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[REPORT] Execution report written to: {args.json_report}")


if __name__ == "__main__":
    main()
