"""
CSHSM Human Review CLI and Batch Processor.

Processes candidate counter-speech entries from the review queue into either
approved_kb.jsonl (human-approved) or rejected.jsonl (human-rejected).

Guarantees:
- Enforces explicit human reviewer ID and human_audited=True.
- Validates each entry against KBEntry schema before appending.
- Maintains strict auditability and provenance.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.kb.schema import KBEntry, validate_entry

DEFAULT_QUEUE = "data/counter_speech_kb/review_queue.jsonl"
DEFAULT_APPROVED = "data/counter_speech_kb/approved_kb.jsonl"
DEFAULT_REJECTED = "data/counter_speech_kb/rejected.jsonl"


def load_queue(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        return []
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_queue(filepath: str, queue: List[Dict[str, Any]]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for c in queue:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def append_to_file(filepath: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def apply_decision(
    candidate: Dict[str, Any],
    decision: str,
    reviewer_id: str,
    notes: str = ""
) -> Dict[str, Any]:
    """Updates candidate metadata with genuine human audit fields."""
    c = dict(candidate)
    now_iso = datetime.now(timezone.utc).isoformat()

    c["human_audited"] = True
    c["review_status"] = "human-approved" if decision == "a" else "human-rejected"
    c["quality_status"] = "approved" if decision == "a" else "rejected"
    c["notes"] = f"Human Review ({reviewer_id}): {notes}".strip()
    c["review_score"] = {"human_decision": 5 if decision == "a" else 1}
    c["created_at"] = now_iso

    # Validate against formal schema
    entry_obj = KBEntry(**c)
    if not validate_entry(entry_obj):
        raise ValueError(f"Entry {c.get('kb_id')} failed schema validation after review.")

    return c


def main():
    parser = argparse.ArgumentParser(description="Human Review CLI and Batch Processor for CSHSM KB")
    parser.add_argument("--queue", type=str, default=DEFAULT_QUEUE, help="Path to input review queue JSONL.")
    parser.add_argument("--approved", type=str, default=DEFAULT_APPROVED, help="Path to output approved KB JSONL.")
    parser.add_argument("--rejected", type=str, default=DEFAULT_REJECTED, help="Path to output rejected JSONL.")
    parser.add_argument("--reviewer-id", type=str, required=True, help="Mandatory identifier for human reviewer.")
    parser.add_argument("--decisions-file", type=str, default=None, help="Optional JSON file mapping kb_id -> 'a'|'r' for batch review.")
    args = parser.parse_args()

    queue = load_queue(args.queue)
    if not queue:
        print(f"Review queue at {args.queue} is empty or not found.")
        return

    print("=" * 60)
    print(f"CSHSM Human Review Session — Reviewer: {args.reviewer_id}")
    print(f"Queue: {args.queue} ({len(queue)} items pending)")
    print("=" * 60)

    # Batch decision mode via decisions file
    if args.decisions_file:
        if not os.path.exists(args.decisions_file):
            raise FileNotFoundError(f"Decisions file not found: {args.decisions_file}")
        with open(args.decisions_file, "r", encoding="utf-8") as f:
            decisions_map = json.load(f)

        remaining = []
        approved_cnt = 0
        rejected_cnt = 0

        for c in queue:
            kb_id = c.get("kb_id")
            if kb_id in decisions_map:
                dec = decisions_map[kb_id]
                action = dec if isinstance(dec, str) else dec.get("action", "s")
                notes = "" if isinstance(dec, str) else dec.get("notes", "")

                if action.lower() in ["a", "approve"]:
                    reviewed = apply_decision(c, "a", args.reviewer_id, notes)
                    append_to_file(args.approved, reviewed)
                    approved_cnt += 1
                elif action.lower() in ["r", "reject"]:
                    reviewed = apply_decision(c, "r", args.reviewer_id, notes)
                    append_to_file(args.rejected, reviewed)
                    rejected_cnt += 1
                else:
                    remaining.append(c)
            else:
                remaining.append(c)

        save_queue(args.queue, remaining)
        print(f"Batch review applied: {approved_cnt} approved, {rejected_cnt} rejected, {len(remaining)} remaining.")
        return

    # Interactive mode
    remaining_queue = []
    for i, c in enumerate(queue):
        print("\n" + "-" * 50)
        print(f"Candidate {i + 1} of {len(queue)} | ID: {c.get('kb_id')} | Category: {c.get('category')} ({c.get('target')})")
        print("-" * 50)
        print(f"HATE TEXT:\n  {c.get('hate_text')}")
        print(f"COUNTER-SPEECH:\n  {c.get('counter_speech')}")
        print("-" * 50)

        while True:
            try:
                decision = input("Decision [Approve (a) / Reject (r) / Skip (s) / Quit (q)]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                decision = "q"
            if decision in ["a", "r", "s", "q"]:
                break
            print("Invalid input. Please enter 'a', 'r', 's', or 'q'.")

        if decision == "q":
            remaining_queue.extend(queue[i:])
            print("\nExiting review session.")
            break
        elif decision == "s":
            remaining_queue.append(c)
            continue

        try:
            notes = input("Optional notes: ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""

        if decision == "a":
            reviewed = apply_decision(c, "a", args.reviewer_id, notes)
            append_to_file(args.approved, reviewed)
            print(f"-> Candidate {c.get('kb_id')} APPROVED.")
        else:
            reviewed = apply_decision(c, "r", args.reviewer_id, notes)
            append_to_file(args.rejected, reviewed)
            print(f"-> Candidate {c.get('kb_id')} REJECTED.")

    save_queue(args.queue, remaining_queue)
    print(f"\nReview session complete. {len(remaining_queue)} candidates remain in queue.")


if __name__ == "__main__":
    main()
