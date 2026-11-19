"""
CSHSM Phase 2: Demonstration Knowledge Base Construction & Semantic Indexing.

Objectives:
1. Generate diverse, real AI counter-speech candidates across all 6 trained categories:
   - Gender
   - Political
   - Religion
   - Geo-political
   - Violence
   - Others
2. Use Google Gemini (gemini-flash-lite-latest) via free API for genuine generation.
3. Apply automated quality, structural, leakage, and safety filters.
4. Evaluate candidates using Google Gemini as an LLM judge (1-5 rubrics).
5. Route approved entries to demo_approved_kb.jsonl (marked ai_generated, automatically_validated, human_audited=False).
6. Route rejected entries to demo_rejected.jsonl with exact reasons.
7. Run EmbeddingPipeline and build vector index.
8. Validate retrieval queries across categories.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(".env")

from src.kb.schema import KBEntry, validate_entry
from src.kb.filters import safety_filter, leakage_filter
from src.retrieval.embedder import MockEmbedder, EmbeddingPipeline
from src.retrieval.retriever import CSHSMRetriever, ExactInnerProductIndex, create_vector_index


# Reference datasets for leakage checking
def load_leakage_sets() -> Tuple[set, set]:
    test_set = set()
    val_set = set()
    test_path = "data/splits/test.jsonl"
    val_path = "data/splits/val.jsonl"
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    test_set.add(json.loads(l).get("Comment", "").strip().lower())
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    val_set.add(json.loads(l).get("Comment", "").strip().lower())
    return test_set, val_set


# Input Prompts: 4 diverse contexts per category across all 6 CSHSM categories
PROMPTS = [
    # 1. Gender
    {"category": "Gender", "target": "Women", "hate_text": "Women do not belong in leadership or engineering, their brains are built only for housework."},
    {"category": "Gender", "target": "LGBT+", "hate_text": "Transgender individuals are mentally ill degenerates who should be removed from public society."},
    {"category": "Gender", "target": "Women", "hate_text": "Educating daughters is a complete waste of family money because they only belong in their husband's kitchen."},
    {"category": "Gender", "target": "Women", "hate_text": "Working women are destroying traditional culture and moral values in our country."},

    # 2. Political
    {"category": "Political", "target": "Opposition", "hate_text": "Anyone voting for the opposition party is a traitor and should lose their right to citizenship."},
    {"category": "Political", "target": "Activists", "hate_text": "Student activists protesting against the ruling party should be beaten and locked away indefinitely."},
    {"category": "Political", "target": "Liberals", "hate_text": "These corrupt liberal intellectuals are parasites actively plotting to dismantle our civilization."},
    {"category": "Political", "target": "Dissidents", "hate_text": "Those criticizing government policy are foreign agents who deserve no legal protection or human rights."},

    # 3. Religion
    {"category": "Religion", "target": "Muslims", "hate_text": "All Muslims are violent radicals who are secretly planning to wage war on our homeland."},
    {"category": "Religion", "target": "Hindus", "hate_text": "Hindus are polytheistic idol-worshippers whose backward customs have corrupted modern society."},
    {"category": "Religion", "target": "Minorities", "hate_text": "Religious minorities are a dangerous virus threatening the purity of our historic faith."},
    {"category": "Religion", "target": "Jews", "hate_text": "Jews control global finance and media and are secretly orchestrating crises worldwide."},

    # 4. Geo-political
    {"category": "Geo-political", "target": "Immigrants", "hate_text": "Immigrants and refugees are dirty parasites who bring disease, crime, and drain our taxes."},
    {"category": "Geo-political", "target": "Migrants", "hate_text": "People from neighboring regions are thieves and freeloaders who have ruined our beautiful city."},
    {"category": "Geo-political", "target": "Foreigners", "hate_text": "Foreign workers are stealing jobs from native citizens and should be rounded up and deported immediately."},
    {"category": "Geo-political", "target": "Refugees", "hate_text": "Refugees crossing the border should be met with armed military force and expelled without mercy."},

    # 5. Violence
    {"category": "Violence", "target": "Protestors", "hate_text": "Those peaceful protestors should be run over with bulldozers so they never dare demonstrate again."},
    {"category": "Violence", "target": "Criminals", "hate_text": "We must form armed vigilante mobs and lynch these criminals publicly instead of relying on courts."},
    {"category": "Violence", "target": "Communities", "hate_text": "The only way to achieve lasting peace is to completely eliminate and slaughter that entire community."},
    {"category": "Violence", "target": "Opponents", "hate_text": "Violence is the only language these subhuman animals understand; peaceful talk is for cowards."},

    # 6. Others
    {"category": "Others", "target": "Lower-caste", "hate_text": "Lower-caste people are inherently inferior and should never be granted university seats or professional jobs."},
    {"category": "Others", "target": "Disabled", "hate_text": "People with cognitive and physical disabilities are useless burdens consuming resources meant for productive citizens."},
    {"category": "Others", "target": "Race", "hate_text": "People with darker skin are biologically less civilized and naturally suited only for menial labor."},
    {"category": "Others", "target": "Caste", "hate_text": "Traditional caste hierarchy must be strictly enforced to keep untouchables in their rightful subordinate place."}
]


def call_gemini_generation(api_key: str, category: str, hate_text: str, timeout: float = 30.0) -> str:
    """Invokes Google Gemini flash-lite to generate an empathetic counter-speech response."""
    system_prompt = (
        "You are an expert researcher in counter-speech generation for mitigating online toxicity. "
        "Your task: generate a constructive, de-escalating, empathetic, and factual counter-narrative "
        "that dismantles prejudice without insulting the speaker or using toxic slurs. "
        "Keep the response concise (1 to 3 sentences, between 30 and 350 characters)."
    )
    user_prompt = f"Category: {category}\nHate Speech Context: {hate_text}\nGenerate counter-speech:"

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200
        }
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
    return ""


def call_gemini_evaluator(api_key: str, hate_text: str, counter_speech: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Invokes Google Gemini as an LLM-as-a-judge to evaluate the candidate on a 1-5 integer scale."""
    eval_prompt = f"""You are an expert AI judge evaluating a counter-speech response for social media toxicity research.

Hate Text: "{hate_text}"
Counter-Speech Candidate: "{counter_speech}"

Rate the counter-speech candidate on a 1-5 integer scale across these four criteria:
1. relevance (1-5): How directly and accurately it addresses the context.
2. persuasiveness (1-5): How compelling, factual, and logically sound the argument is.
3. de_escalation (1-5): How effectively it avoids hostility and promotes calm constructive dialogue.
4. safety (1-5): 5 if completely safe, non-toxic, non-threatening; 1 if harmful or hateful.

Respond with ONLY valid JSON with keys:
"relevance", "persuasiveness", "de_escalation", "safety", "reason"
"""
    payload = json.dumps({
        "contents": [{"parts": [{"text": eval_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 150
        }
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Clean markdown codeblocks if returned
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())


def call_with_retry(func, *args, max_retries: int = 6, initial_wait: float = 20.0, **kwargs):
    """Executes func with exponential backoff on HTTP 429 rate limit errors."""
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    [HTTP 429 rate limit. Backing off for {wait:.1f}s (attempt {attempt+1}/{max_retries})...]")
                time.sleep(wait)
                wait *= 1.5
            else:
                raise
        except Exception as e:
            if "429" in str(e):
                print(f"    [Rate limit 429. Backing off for {wait:.1f}s (attempt {attempt+1}/{max_retries})...]")
                time.sleep(wait)
                wait *= 1.5
            else:
                raise
    raise RuntimeError(f"API call failed after {max_retries} attempts due to rate limiting.")


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY is not set in .env. Cannot perform genuine generation.")
        sys.exit(1)

    print("=" * 80)
    print("CSHSM Phase 2: Demonstration KB Generation & Evaluation")
    print("=" * 80)

    test_leakage, val_leakage = load_leakage_sets()
    print(f"Loaded leakage reference sets: Test={len(test_leakage)}, Val={len(val_leakage)}")

    approved_entries: List[KBEntry] = []
    rejected_entries: List[Dict[str, Any]] = []

    os.makedirs("data/counter_speech_kb", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Check for existing approved records to avoid redundant API calls
    existing_approved: Dict[str, KBEntry] = {}
    approved_path = "data/counter_speech_kb/demo_approved_kb.jsonl"
    if os.path.exists(approved_path):
        with open(approved_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    try:
                        rec = json.loads(l)
                        existing_approved[rec["hate_text"].strip()] = KBEntry(**rec)
                    except Exception:
                        pass
    print(f"Found {len(existing_approved)} existing approved demo KB entries on disk.")

    start_time = time.perf_counter()

    for idx, item in enumerate(PROMPTS):
        category = item["category"]
        target = item["target"]
        hate_text = item["hate_text"]
        kb_id = f"DEMO-KB-{idx+1:05d}"

        # If already generated and approved, reuse!
        if hate_text.strip() in existing_approved:
            e = existing_approved[hate_text.strip()]
            approved_entries.append(e)
            print(f"\n[{idx+1}/{len(PROMPTS)}] Category: {category} ({target}) -> REUSING APPROVED ENTRY ({e.kb_id})")
            continue

        print(f"\n[{idx+1}/{len(PROMPTS)}] Category: {category} ({target})")
        print(f"Hate text: \"{hate_text[:70]}...\"")

        # 1. Leakage Check
        is_no_leak, leak_reason = leakage_filter(hate_text, test_leakage, val_leakage)
        if not is_no_leak:
            print(f" -> REJECTED: {leak_reason}")
            rejected_entries.append({
                "kb_id": kb_id,
                "category": category,
                "hate_text": hate_text,
                "rejection_stage": "leakage_filter",
                "rejection_reason": leak_reason
            })
            continue

        # 2. Real AI Generation
        t_gen_start = time.perf_counter()
        try:
            counter_speech = call_with_retry(call_gemini_generation, api_key, category, hate_text)
            gen_latency_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            print(f" -> Generated in {gen_latency_ms} ms: \"{counter_speech[:70]}...\"")
        except Exception as e:
            print(f" -> GENERATION FAILED: {e}")
            rejected_entries.append({
                "kb_id": kb_id,
                "category": category,
                "hate_text": hate_text,
                "rejection_stage": "generation_api",
                "rejection_reason": f"API Error: {e}"
            })
            continue

        # Courteous pacing
        time.sleep(4.0)

        # 3. Structural Validation
        char_len = len(counter_speech)
        if char_len < 20 or char_len > 500:
            reason = f"Length out of bounds ({char_len} chars)"
            print(f" -> REJECTED: {reason}")
            rejected_entries.append({
                "kb_id": kb_id,
                "category": category,
                "hate_text": hate_text,
                "counter_speech": counter_speech,
                "rejection_stage": "structural_validation",
                "rejection_reason": reason
            })
            continue

        # 4. Safety Filter Check
        is_safe, safety_reason = safety_filter(counter_speech)
        if not is_safe:
            print(f" -> REJECTED: Safety violation ({safety_reason})")
            rejected_entries.append({
                "kb_id": kb_id,
                "category": category,
                "hate_text": hate_text,
                "counter_speech": counter_speech,
                "rejection_stage": "safety_filter",
                "rejection_reason": safety_reason
            })
            continue

        # 5. Real LLM-as-a-judge Evaluation
        t_eval_start = time.perf_counter()
        try:
            eval_result = call_with_retry(call_gemini_evaluator, api_key, hate_text, counter_speech)
            eval_latency_ms = round((time.perf_counter() - t_eval_start) * 1000, 2)
            scores = {
                "relevance": int(eval_result.get("relevance", 0)),
                "persuasiveness": int(eval_result.get("persuasiveness", 0)),
                "de_escalation": int(eval_result.get("de_escalation", 0)),
                "safety": int(eval_result.get("safety", 0))
            }
            eval_reason = str(eval_result.get("reason", ""))
            print(f" -> Evaluated in {eval_latency_ms} ms: Scores={scores}")
        except Exception as e:
            print(f" -> EVALUATION FAILED: {e}")
            rejected_entries.append({
                "kb_id": kb_id,
                "category": category,
                "hate_text": hate_text,
                "counter_speech": counter_speech,
                "rejection_stage": "evaluation_api",
                "rejection_reason": f"Evaluation API Error: {e}"
            })
            continue

        # Courteous pacing
        time.sleep(4.0)

        # 6. Evaluation Threshold Gating
        if scores["safety"] < 4 or scores["relevance"] < 3 or scores["de_escalation"] < 3:
            reason = f"Evaluation score below threshold: {scores}"
            print(f" -> REJECTED: {reason}")
            rejected_entries.append({
                "kb_id": kb_id,
                "category": category,
                "hate_text": hate_text,
                "counter_speech": counter_speech,
                "scores": scores,
                "rejection_stage": "evaluation_threshold",
                "rejection_reason": reason
            })
            continue

        # 7. Candidate Accepted into Demonstration KB
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = KBEntry(
            kb_id=kb_id,
            hate_text=hate_text,
            counter_speech=counter_speech,
            category=category,
            target=target,
            strategy="empathy_and_factual_reframing",
            language="en",
            source="ai_demonstration_corpus",
            source_id="gemini-flash-lite-latest",
            provenance="ai_generated",
            generation_method="gemini-flash-lite-latest",
            quality_status="automatically_validated",
            review_status="automatically_validated",
            review_score=scores,
            human_audited=False,
            created_at=now_iso,
            version="1.0",
            notes=f"AI-generated demonstration candidate; automatically validated by Gemini judge ({eval_reason})"
        )

        assert validate_entry(entry), f"Schema validation failed for {kb_id}"
        approved_entries.append(entry)
        print(" -> ACCEPTED into Demonstration KB.")

    total_time = round(time.perf_counter() - start_time, 2)

    # Persist JSONL files
    approved_path = "data/counter_speech_kb/demo_approved_kb.jsonl"
    rejected_path = "data/counter_speech_kb/demo_rejected.jsonl"

    with open(approved_path, "w", encoding="utf-8") as f:
        for e in approved_entries:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in rejected_entries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print("DEMONSTRATION KB SUMMARY")
    print("=" * 80)
    print(f"Total Candidates Processed: {len(PROMPTS)}")
    print(f"Total Accepted: {len(approved_entries)} -> {approved_path}")
    print(f"Total Rejected: {len(rejected_entries)} -> {rejected_path}")
    print(f"Total Processing Time: {total_time}s")

    # Category breakdown
    cat_counts = {}
    for e in approved_entries:
        cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
    print(f"Category Distribution: {cat_counts}")

    # Build Embedding Index using EmbeddingPipeline
    print("\n" + "=" * 80)
    print("BUILDING SEMANTIC EMBEDDING INDEX")
    print("=" * 80)

    # Run EmbeddingPipeline in development_provisional scope on demo_approved_kb
    embedder = MockEmbedder(dim=1024)
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="development_provisional",
        output_dir="data/counter_speech_kb/embeddings"
    )
    run_meta = pipeline.run(approved_path)
    print(f"Committed embedding run: {run_meta['run_id']}")
    print(f"Vectors file: {run_meta['artifacts']['vector_file']}")
    print(f"Latest pointer: {run_meta['artifacts']['latest_pointer']}")

    # Test CSHSMRetriever on the committed run
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL OVER DEMONSTRATION KB")
    print("=" * 80)

    retriever = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=True,
        embedder=embedder
    )
    print(f"Retriever initialized. Total indexed items: {retriever.index.ntotal}")
    print(f"Retriever backend: {retriever.index.backend_name}")

    test_queries = [
        ("Religion", "Muslims are dangerous radicals threatening peace."),
        ("Gender", "Women are biologically unfit for hard technical careers."),
        ("Political", "Opposition politicians are unpatriotic foreign stooges."),
        ("Violence", "We must lynch these criminals without mercy.")
    ]

    retrieval_demonstrations = []
    for cat, q in test_queries:
        res = retriever.retrieve_by_text(q, top_k=2, category_filter=None)
        print(f"\nQuery ({cat}): \"{q}\"")
        for match in res.matches:
            print(f"  [Rank {match.rank}] Score: {match.score:.4f} | Cat: {match.category} | ID: {match.kb_id}")
            print(f"    Match Hate Text: \"{match.hate_text}\"")
            print(f"    Counter-Speech: \"{match.counter_speech}\"")
        retrieval_demonstrations.append({
            "query": q,
            "target_category": cat,
            "top_match_id": res.matches[0].kb_id if res.matches else None,
            "top_match_score": res.matches[0].score if res.matches else None,
            "top_match_counter_speech": res.matches[0].counter_speech if res.matches else None
        })

    # Generate Audit Report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_generated": len(PROMPTS),
        "total_accepted": len(approved_entries),
        "total_rejected": len(rejected_entries),
        "category_distribution": cat_counts,
        "language_distribution": {"en": len(approved_entries)},
        "generation_method": "real_google_gemini_flash_lite",
        "evaluation_method": "real_google_gemini_flash_lite_judge",
        "safety_filtering": "deterministic_safety_and_leakage_filter",
        "human_verified": False,
        "vector_index": {
            "total_indexed": retriever.index.ntotal,
            "dimension": retriever.index.dimension,
            "backend": retriever.index.backend_name,
            "model_identifier": retriever.embedder.model_identifier,
            "is_mock_embedder": retriever.embedder.is_mock,
            "faiss_installed": False,
            "backend_fallback": "ExactInnerProductIndex (pure-Python exact flat inner product)"
        },
        "retrieval_demonstrations": retrieval_demonstrations,
        "execution_time_seconds": total_time
    }

    report_path = "reports/phase2_demo_kb_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nPhase 2 Summary report written to: {report_path}")

if __name__ == "__main__":
    main()
