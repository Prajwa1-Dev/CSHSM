"""
CSHSM Milestone 14 / Phase 3: End-to-End Pipeline Demonstration Script.

Verifies the complete CSHSM pipeline across five critical research scenarios:
1. Non-Hate: Benign comment -> early exit -> no counter-speech.
2. Successful Retrieval: Relevant hate comment -> KB retrieval -> validated retrieved counter-speech.
3. Generation Fallback: Hate comment with no KB match -> router fallback -> 5-model fan-out with real Gemini -> ranked & gate validated counter-speech.
4. Safety Rejection: Unsafe response rejected by defense-in-depth gate -> safe fallback response.
5. Empty Input: Empty/whitespace input handled safely -> fallback without crash.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(".env")

from src.data.cleaner import TextCleaner
from src.pipeline.classifier import XLMROracleClassifier
import torch
from src.retrieval.embedder import MockEmbedder, BGE_M3_Embedder
from src.retrieval.retriever import CSHSMRetriever
from src.routing.router import FallbackRouter
from src.generation.adapters import GoogleGeminiAdapter, MockModelAdapter
from src.generation.fan_out import GenerationFanOut
from src.evaluation.evaluator import MockCandidateEvaluator
from src.evaluation.ranker import CandidateRanker
from src.safety.final_gate import FinalSafetyGate
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineResult


def build_demonstration_pipeline(
    gemini_model_id: str = "gemini-flash-lite-latest",
    threshold: float = 0.80
) -> PipelineOrchestrator:
    print("[1/6] Loading real XLM-R classifier on CUDA...")
    classifier = XLMROracleClassifier(
        model_path="models/xlmr_best.pt",
        device="cuda"
    )
    print(f"      XLM-R loaded. Device: {classifier.device}, Model: {classifier.model_name}")

    print("[2/6] Loading demonstration KB retriever from committed run (BGE-M3 dense embeddings)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device=device,
        local_files_only=True
    )
    kb_output_dir = os.getenv("CSHSM_KB_EMBEDDINGS_DIR")
    if not kb_output_dir:
        kb_output_dir = "data/counter_speech_kb/embeddings_v2" if os.getenv("CSHSM_KB_VERSION", "").strip().lower() in ("v2", "2") else "data/counter_speech_kb/embeddings"
    retriever = CSHSMRetriever.from_committed_run(
        output_dir=kb_output_dir,
        allow_mock=False,
        embedder=embedder,
        force_backend="faiss"
    )
    print(f"      Retriever loaded. Total items indexed: {retriever.index.ntotal}, Backend: {retriever.index.backend_name}")

    print("[3/6] Initializing FallbackRouter with threshold tau = 0.80...")
    router = FallbackRouter(retriever=retriever, default_threshold=threshold)

    print("[4/6] Initializing GenerationFanOut (Slot 3 = Real Gemini, Slots 1,2,4,5 = Mock)...")
    adapters = [
        MockModelAdapter(
            model_id="mock-openai-gpt4o-mini",
            provider="mock",
            canned_response="Respectful and constructive dialogue helps bridge differences rather than widening prejudice."
        ),
        MockModelAdapter(
            model_id="mock-anthropic-claude3-haiku",
            provider="mock",
            canned_response="Promoting hostility towards individuals diminishes our shared community values."
        ),
        GoogleGeminiAdapter(
            model_id=gemini_model_id,
            timeout=30.0
        ),
        MockModelAdapter(
            model_id="mock-mistral-small",
            provider="mock",
            canned_response="Diversity of perspectives is fundamental to a free and flourishing society."
        ),
        MockModelAdapter(
            model_id="mock-cohere-command-r",
            provider="mock",
            canned_response="Every individual deserves dignity, safety, and equal consideration regardless of identity."
        )
    ]
    fan_out = GenerationFanOut(adapters=adapters, overall_timeout=70.0)

    print("[5/6] Initializing Evaluator, Ranker, and Final Safety Gate...")
    # Give real Gemini adapter higher score than default mocks to reward genuine generation
    evaluator = MockCandidateEvaluator(
        candidate_overrides={
            gemini_model_id: {
                "relevance": 5,
                "persuasiveness": 5,
                "de_escalation": 5,
                "safety": 5
            }
        }
    )
    ranker = CandidateRanker(evaluator=evaluator)
    final_gate = FinalSafetyGate()
    cleaner = TextCleaner()

    print("[6/6] Assembling PipelineOrchestrator...")
    orchestrator = PipelineOrchestrator(
        classifier=classifier,
        router=router,
        fan_out=fan_out,
        ranker=ranker,
        final_gate=final_gate,
        cleaner=cleaner,
        allow_mock=True,
        threshold=threshold
    )
    return orchestrator


def run_scenarios(orchestrator: PipelineOrchestrator) -> List[Dict[str, Any]]:
    scenarios = [
        # Scenario 1: Non-Hate
        {
            "id": 1,
            "name": "Non-hate early exit",
            "input": "Wishing everyone a peaceful and productive week ahead! Let's collaborate together.",
            "description": "Benign positive statement; classifier must classify as NON-HATE and exit early without generating counter-speech."
        },
        # Scenario 2: Successful Retrieval
        {
            "id": 2,
            "name": "Successful KB retrieval",
            "input": "Women do not belong in leadership or engineering, their brains are built only for housework.",
            "description": "Exact match to DEMO-KB-00001; classifier predicts Gender hate; router detects similarity >= 0.80 and retrieves from KB."
        },
        # Scenario 3: Generation Fallback
        {
            "id": 3,
            "name": "Generation fan-out fallback",
            "input": "Every member of that political party is corrupt and should be thrown into prison without trial.",
            "description": "Novel political hate speech not in KB; similarity < 0.80 triggers fan-out; real Gemini in Slot 3 generates live response."
        },
        # Scenario 4: Safety Rejection
        {
            "id": 4,
            "name": "Safety rejection and safe fallback",
            "input": "TEST_SAFETY_REJECTION_TRIGGER",  # Handled with custom unsafe adapter injection or candidate test
            "description": "Candidate containing prohibited or toxic text is caught by defense-in-depth safety gate and triggers safe fallback."
        },
        # Scenario 5: Empty Input
        {
            "id": 5,
            "name": "Empty input safe handling",
            "input": "    ",
            "description": "Whitespace only input; cleaner catches empty string and returns safe fallback without classifier crash."
        }
    ]

    results = []

    print("\n" + "=" * 80)
    print("RUNNING 5 DEMONSTRATION SCENARIOS")
    print("=" * 80)

    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        inp = sc["input"]
        print(f"\n--- Scenario {sc_id}: {sc_name} ---")
        print(f"Input: \"{inp}\"")

        t0 = time.perf_counter()

        if sc_id == 4:
            # For scenario 4, demonstrate safety rejection:
            # We configure a pipeline where the candidate is rejected by the safety filter
            print("  [Configuring test unsafe fan-out candidate to demonstrate safety filter rejection]")
            unsafe_adapters = [
                MockModelAdapter(
                    model_id="mock-unsafe-model",
                    provider="mock",
                    canned_response="Shut up you idiot, go to hell."  # Trigger: counter-hate pattern in safety_filter
                )
            ]
            unsafe_fan_out = GenerationFanOut(adapters=unsafe_adapters)
            unsafe_orchestrator = PipelineOrchestrator(
                classifier=orchestrator.classifier,
                router=FallbackRouter(retriever=orchestrator.router.retriever, default_threshold=0.99),  # force fallback to generation
                fan_out=unsafe_fan_out,
                ranker=orchestrator.ranker,
                final_gate=orchestrator.final_gate,
                cleaner=orchestrator.cleaner,
                allow_mock=True,
                threshold=0.99
            )
            # Use a hate input that routes to generation
            test_input = "All members of that minority are evil subhumans who must be expelled."
            res: PipelineResult = unsafe_orchestrator.process(test_input)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            actual_input = test_input
        else:
            actual_input = inp
            res: PipelineResult = orchestrator.process(inp)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        print(f"Outcome: {res.outcome}")
        print(f"Is Hate: {res.is_hate} | Category: {res.category} | Confidence: {res.confidence:.4f}")
        print(f"Source Origin: {res.source_origin} | Candidate ID: {res.candidate_id} | Provider: {res.provider}")
        print(f"Safe Fallback: {res.safe_fallback_triggered} | Reason: {res.fallback_reason}")
        print(f"Response: \"{res.response_text or res.fallback_response}\"")
        print(f"Latency: {latency_ms} ms")

        results.append({
            "scenario_id": sc_id,
            "name": sc_name,
            "description": sc["description"],
            "input_text": actual_input,
            "outcome": res.outcome,
            "is_hate": res.is_hate,
            "category": res.category,
            "confidence": res.confidence,
            "source_origin": res.source_origin,
            "candidate_id": res.candidate_id,
            "provider": res.provider,
            "model_id": res.model_id,
            "quality_status": res.quality_status,
            "safe_fallback_triggered": res.safe_fallback_triggered,
            "fallback_reason": res.fallback_reason,
            "response_text": res.response_text,
            "fallback_response": res.fallback_response,
            "component_statuses": res.component_statuses,
            "latency_ms": latency_ms
        })

    return results


def main():
    print("=" * 80)
    print("CSHSM Phase 3: End-to-End Pipeline Integration & Verification")
    print("=" * 80)

    orchestrator = build_demonstration_pipeline()
    results = run_scenarios(orchestrator)

    # Save results to reports
    out_path = "reports/phase3_demonstration_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classifier": {
                "model_path": "models/xlmr_best.pt",
                "device": orchestrator.classifier.device,
                "categories": orchestrator.classifier.CATEGORIES,
                "is_mock": False
            },
            "retriever": {
                "source": "data/counter_speech_kb/demo_approved_kb.jsonl",
                "embedding_run": orchestrator.router.retriever.run_metadata.get("run_id"),
                "indexed_count": orchestrator.router.retriever.index.ntotal,
                "backend": orchestrator.router.retriever.index.backend_name,
                "is_mock_embedder": orchestrator.router.retriever.embedder.is_mock
            },
            "generation": {
                "live_adapter": "GoogleGeminiAdapter (gemini-flash-lite-latest)",
                "mock_adapters": ["mock-openai-gpt4o-mini", "mock-anthropic-claude3-haiku", "mock-mistral-small", "mock-cohere-command-r"],
                "total_slots": 5
            },
            "scenarios": results
        }, f, indent=2)

    print(f"\nPhase 3 demonstration report written to: {out_path}")


if __name__ == "__main__":
    main()
