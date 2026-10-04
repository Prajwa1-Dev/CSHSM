import os
import json
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.kb.schema import KBEntry

DATA_DIR = "data/counter_speech_kb"
REPORTS_DIR = "reports"

def check_llm_availability():
    """
    Check if a real LLM endpoint or local model is configured for evaluation.
    We look for standard API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) or a local config flag.
    """
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LOCAL_LLM_EVAL_READY"):
        return True
    return False

def load_candidates():
    candidates = []
    
    # 1. Load M5.2 filtered candidates
    m52_path = os.path.join(DATA_DIR, "filtered_candidates.jsonl")
    if os.path.exists(m52_path):
        with open(m52_path, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                candidates.append(KBEntry(**c))
                
    # 2. Load M5.3 LLM candidates
    m53_path = os.path.join(DATA_DIR, "m53_generation", "llm_generated_candidates.jsonl")
    if os.path.exists(m53_path):
        with open(m53_path, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                candidates.append(KBEntry(**c))
                
    return candidates

def main():
    has_llm = check_llm_availability()
    
    candidates = load_candidates()
    
    human_review_required = []
    eligible_deferred = [] # Not used unless evaluation passes but capped by diversity
    approved_kb = []
    rejected = []
    
    stats = {
        "m52": {"input": 0, "approved": 0, "human_review_required": 0, "rejected": 0, "eligible_deferred": 0, "blocked_eval": 0},
        "m53": {"input": 0, "approved": 0, "human_review_required": 0, "rejected": 0, "eligible_deferred": 0, "blocked_eval": 0}
    }
    
    for c in candidates:
        is_m53 = (c.provenance == "llm-assisted")
        stat_key = "m53" if is_m53 else "m52"
        stats[stat_key]["input"] += 1
        
        if is_m53:
            # Mandate: LLM-assisted candidates REQUIRE actual human review.
            c.quality_status = "human_review_required"
            c.notes = "LLM-assisted candidate: Mandatory human review required."
            human_review_required.append(c)
            stats[stat_key]["human_review_required"] += 1
        else:
            # Source-derived candidates require real LLM evaluation to proceed.
            if not has_llm:
                c.quality_status = "blocked_eval"
                stats[stat_key]["blocked_eval"] += 1
            else:
                # Placeholder for real LLM evaluation logic if an API key was present.
                pass
                
    # Write artifacts
    with open(os.path.join(DATA_DIR, "human_review_required.jsonl"), 'w', encoding='utf-8') as f:
        for c in human_review_required:
            f.write(json.dumps(c.to_dict()) + "\n")
            
    # Initialize empty files to satisfy artifacts expectations but clearly separated from the simulated ones
    with open(os.path.join(DATA_DIR, "approved_kb_v2.jsonl"), 'w', encoding='utf-8') as f:
        pass
    with open(os.path.join(DATA_DIR, "rejected_v2.jsonl"), 'w', encoding='utf-8') as f:
        pass
    with open(os.path.join(DATA_DIR, "eligible_deferred.jsonl"), 'w', encoding='utf-8') as f:
        pass

    report = {
        "blocker": "No real LLM evaluator available (API keys or local model config missing).",
        "accounting": stats,
        "evaluations_real": 0,
        "evaluations_simulated_in_v2": 0,
        "human_review_actual": 0,
        "human_review_pending": len(human_review_required),
        "final_approved_count": 0,
        "audit_results": "Blocked (No candidates auto-approved to audit)"
    }
    
    with open(os.path.join(REPORTS_DIR, "m5_4_evaluation_v2.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    with open(os.path.join("docs", "M5.4_COMPLETION_REPORT_V2.md"), 'w', encoding='utf-8') as f:
        f.write("# M5.4 Evaluation & State Transitions Report (Corrected)\n\n")
        f.write("## Critical Blocker: Real LLM Evaluation Unavailable\n")
        f.write("The environment lacks the necessary API keys or authorized local model configuration to perform a **real** LLM-as-a-judge evaluation over the 19,436 source-derived candidates. In accordance with research validity requirements, this evaluation has NOT been simulated. The source candidates remain in a `blocked_eval` state.\n\n")
        f.write("## Previous Simulated Artifacts\n")
        f.write("All prior M5.4 artifacts (e.g., `approved_kb_simulated.jsonl`) were based on simulated LLM scores and simulated human review. They have been preserved for engineering validation but are **explicitly marked as simulated** and hold no research validity for the final KB.\n\n")
        f.write("## LLM-Assisted Candidates & Human Review\n")
        f.write("The 18 LLM-generated candidates (M5.3) have been correctly routed to `data/counter_speech_kb/human_review_required.jsonl`. They have **not** been auto-approved. A CLI tool (`scripts/human_reviewer.py`) has been provided to allow actual human review of these candidates.\n\n")
        f.write("## Accounting\n")
        f.write(f"- Real evaluations performed: 0\n")
        f.write(f"- Simulated evaluations in this pass: 0\n")
        f.write(f"- Candidates requiring actual human review: {len(human_review_required)} (All M5.3)\n")
        f.write(f"- Source candidates blocked awaiting LLM evaluator: {stats['m52']['blocked_eval']}\n")
        f.write(f"- Deferred for diversity (`eligible_deferred`): 0 (blocked upstream)\n")
        f.write(f"- Final approved count: 0\n")

if __name__ == "__main__":
    main()
