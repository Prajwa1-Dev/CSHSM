import os
import json
import argparse
from datetime import datetime

def load_queue(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_queue(filepath, queue):
    with open(filepath, 'w', encoding='utf-8') as f:
        for c in queue:
            f.write(json.dumps(c) + "\n")

def append_to_file(filepath, entry):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Human Review CLI for M5.4 Candidates")
    parser.add_argument("--queue", type=str, default="data/counter_speech_kb/human_review_required.jsonl")
    parser.add_argument("--approved", type=str, default="data/counter_speech_kb/approved_kb_v2.jsonl")
    parser.add_argument("--rejected", type=str, default="data/counter_speech_kb/rejected_v2.jsonl")
    parser.add_argument("--reviewer-id", type=str, required=True, help="Identifier for the human reviewer")
    args = parser.parse_args()

    queue = load_queue(args.queue)
    if not queue:
        print("Review queue is empty.")
        return

    print(f"Found {len(queue)} candidates pending human review.")
    
    remaining_queue = []
    
    for i, c in enumerate(queue):
        print("\n" + "="*50)
        print(f"Candidate {i+1} of {len(queue)}")
        print(f"ID: {c.get('kb_id')}")
        print(f"Provenance: {c.get('provenance')}")
        print(f"Category: {c.get('category')}")
        print("-" * 50)
        print(f"HATE TEXT:\n{c.get('hate_text')}")
        print("-" * 50)
        print(f"COUNTER-SPEECH:\n{c.get('counter_speech')}")
        print("="*50)
        
        while True:
            decision = input("Approve (a) / Reject (r) / Skip (s) / Quit (q): ").strip().lower()
            if decision in ['a', 'r', 's', 'q']:
                break
            print("Invalid input.")
            
        if decision == 'q':
            remaining_queue.extend(queue[i:])
            print("Exiting review session.")
            break
        elif decision == 's':
            remaining_queue.append(c)
            continue
            
        reason = input("Optional reason/notes: ").strip()
        
        c['human_audited'] = True
        c['review_status'] = "human-approved" if decision == 'a' else "human-rejected"
        c['quality_status'] = "approved" if decision == 'a' else "rejected"
        c['notes'] = f"Human Review ({args.reviewer_id}): {reason}" if reason else f"Human Review ({args.reviewer_id}): {'Approved' if decision == 'a' else 'Rejected'}"
        
        # We don't have a specific human review score schema, but we can set it to perfect if approved.
        if decision == 'a':
            c['review_score'] = {"human_decision": "pass"}
            append_to_file(args.approved, c)
            print("-> Candidate APPROVED.")
        else:
            c['review_score'] = {"human_decision": "fail"}
            append_to_file(args.rejected, c)
            print("-> Candidate REJECTED.")

    save_queue(args.queue, remaining_queue)
    print(f"\nReview session complete. {len(remaining_queue)} items remain in the queue.")

if __name__ == "__main__":
    main()
