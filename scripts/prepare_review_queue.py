"""
CSHSM Milestone 12.1: Review Queue Preparation Script.

Extracts a curated, diverse, high-quality sample of 25-30 candidate pairs
from data/counter_speech_kb/filtered_candidates.jsonl for human review.

Guarantees:
- Preserves all source fields, IDs, categories, targets, and provenance.
- Excludes duplicate hate texts or duplicate counter-speech texts.
- Enforces strict safety filter (safety_filter) and quality length bounds.
- Explicitly marks every record as pending human review:
    quality_status: "pending_review"
    review_status: "pending_review"
    human_audited: False
- Does NOT auto-approve or fabricate any human review.
"""

import os
import sys
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.kb.schema import KBEntry, validate_entry
from src.kb.filters import safety_filter

INPUT_FILE = "data/counter_speech_kb/filtered_candidates.jsonl"
OUTPUT_QUEUE_FILE = "data/counter_speech_kb/review_queue.jsonl"
REPORT_FILE = "reports/m12_1_review_queue_summary.json"

# Quota definition ensuring balanced coverage across categories and sub-targets
TARGET_QUOTAS: Dict[Tuple[str, str], int] = {
    ("Religion", "Islamophobia"): 3,
    ("Religion", "MUSLIMS"): 2,
    ("Religion", "JEWS"): 5,
    ("Gender", "WOMEN"): 4,
    ("Gender", "LGBT+"): 4,
    ("Geo-political", "MIGRANTS"): 5,
    ("Others", "POC"): 5
}


def prepare_review_queue(
    input_path: str = INPUT_FILE,
    output_path: str = OUTPUT_QUEUE_FILE,
    report_path: str = REPORT_FILE
) -> List[Dict]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source candidate dataset not found at: {input_path}")

    counts: Dict[Tuple[str, str], int] = {k: 0 for k in TARGET_QUOTAS}
    selected_entries: List[KBEntry] = []
    seen_hate_texts = set()
    seen_counter_texts = set()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            category = record.get("category")
            target = record.get("target")
            key = (category, target)

            if key not in TARGET_QUOTAS:
                continue
            if counts[key] >= TARGET_QUOTAS[key]:
                continue

            hate_text = record.get("hate_text", "").strip()
            counter_speech = record.get("counter_speech", "").strip()

            # Prevent duplicate prompts or repetitive responses
            if hate_text in seen_hate_texts or counter_speech in seen_counter_texts:
                continue

            # Length bounds for high quality demonstration
            if len(hate_text) < 20 or len(counter_speech) < 30 or len(counter_speech) > 400:
                continue

            # Mandatory defense-in-depth safety filter check
            is_safe, rejection_reason = safety_filter(counter_speech)
            if not is_safe:
                continue

            entry = KBEntry(
                kb_id=record["kb_id"],
                hate_text=hate_text,
                counter_speech=counter_speech,
                category=category,
                target=target,
                strategy=record.get("strategy"),
                language=record.get("language", "en"),
                source=record.get("source", "unknown"),
                source_id=str(record.get("source_id", "")),
                provenance=record.get("provenance", "existing-dataset"),
                generation_method=record.get("generation_method", "human-authored"),
                quality_status="pending_review",
                review_status="pending_review",
                review_score=None,
                human_audited=False,
                created_at=record.get("created_at", datetime.now(timezone.utc).isoformat()),
                version="1.0",
                notes="Sampled for M12.1 human review queue"
            )

            if not validate_entry(entry):
                continue

            selected_entries.append(entry)
            seen_hate_texts.add(hate_text)
            seen_counter_texts.add(counter_speech)
            counts[key] += 1

            # Stop early if all quotas are fulfilled
            if all(counts[k] >= TARGET_QUOTAS[k] for k in TARGET_QUOTAS):
                break

    # Verify all quotas were satisfied
    total_sampled = len(selected_entries)
    if total_sampled < sum(TARGET_QUOTAS.values()):
        raise RuntimeError(
            f"Could not fulfill all target quotas. Expected {sum(TARGET_QUOTAS.values())}, got {total_sampled}"
        )

    # Write queue file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in selected_entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # Generate summary report
    category_summary = Counter(e.category for e in selected_entries)
    target_summary = Counter(f"{e.category}:{e.target}" for e in selected_entries)
    source_summary = Counter(e.source for e in selected_entries)

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path,
        "output_file": output_path,
        "total_records_sampled": total_sampled,
        "categories": dict(category_summary),
        "targets": dict(target_summary),
        "sources": dict(source_summary),
        "status": "READY_FOR_HUMAN_REVIEW",
        "human_audited_count": 0,
        "approved_count": 0,
        "pending_review_count": total_sampled
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Successfully generated review queue with {total_sampled} candidates at: {output_path}")
    print(f"Summary report written to: {report_path}")
    return [e.to_dict() for e in selected_entries]


if __name__ == "__main__":
    prepare_review_queue()
