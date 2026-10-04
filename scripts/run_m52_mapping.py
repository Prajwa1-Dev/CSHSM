import os
import csv
import json
import uuid
import re
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.kb.schema import KBEntry, validate_entry
from src.kb.category_mapper import map_category
from src.kb.filters import safety_filter, leakage_filter

DATA_DIR = "data/counter_speech_kb"
SPLITS_DIR = "data/splits"

def load_hastika_texts():
    test_texts = set()
    val_texts = set()
    
    test_path = os.path.join(SPLITS_DIR, "test.jsonl")
    val_path = os.path.join(SPLITS_DIR, "val.jsonl")
    
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if "Comment" in rec:
                    test_texts.add(rec["Comment"].strip().lower())
                    
    if os.path.exists(val_path):
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if "Comment" in rec:
                    val_texts.add(rec["Comment"].strip().lower())
                    
    return test_texts, val_texts

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def process_mt_conan(stats):
    filepath = os.path.join(DATA_DIR, "sources", "Multitarget-CONAN.csv")
    candidates = []
    
    if not os.path.exists(filepath):
        return candidates
        
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["multitarget-conan"]["raw"] += 1
            stats["total"]["raw"] += 1
            
            ht = normalize_text(row.get("HATE_SPEECH", ""))
            cs = normalize_text(row.get("COUNTER_NARRATIVE", ""))
            target = row.get("TARGET", "")
            
            if not ht or not cs:
                stats["multitarget-conan"]["empty"] += 1
                stats["total"]["empty"] += 1
                continue
                
            mapped_cat = map_category("multitarget-conan", target)
            
            entry = KBEntry(
                kb_id="", 
                hate_text=ht,
                counter_speech=cs,
                category=mapped_cat,
                target=target if target else None,
                strategy=None,
                language="en",
                source="multitarget-conan",
                source_id=row.get("INDEX", ""),
                provenance="existing-dataset",
                generation_method="human-authored",
                quality_status="pending",
                review_status="unreviewed",
                review_score=None,
                human_audited=False,
                created_at=datetime.utcnow().isoformat() + "Z",
                version="1.0",
                notes=""
            )
            
            candidates.append(entry)
            
    return candidates

def process_conan(stats):
    filepath = os.path.join(DATA_DIR, "sources", "CONAN.json")
    candidates = []
    
    if not os.path.exists(filepath):
        return candidates
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = data.get("conan", [])
    for row in records:
        stats["conan"]["raw"] += 1
        stats["total"]["raw"] += 1
        
        ht = normalize_text(row.get("hateSpeech", ""))
        cs = normalize_text(row.get("counterSpeech", ""))
        hsType = row.get("hsType", "")
        strategy = row.get("cnType", "")
        
        if not ht or not cs:
            stats["conan"]["empty"] += 1
            stats["total"]["empty"] += 1
            continue
            
        mapped_cat = map_category("conan", hsType)
        
        cn_id = row.get("cn_id", "")
        lang = "en"
        if cn_id.startswith("FR"): lang = "fr"
        elif cn_id.startswith("IT"): lang = "it"
        
        entry = KBEntry(
            kb_id="",
            hate_text=ht,
            counter_speech=cs,
            category=mapped_cat,
            target=hsType if hsType else None,
            strategy=strategy if strategy else None,
            language=lang,
            source="conan",
            source_id=cn_id,
            provenance="existing-dataset",
            generation_method="human-authored",
            quality_status="pending",
            review_status="unreviewed",
            review_score=None,
            human_audited=False,
            created_at=datetime.utcnow().isoformat() + "Z",
            version="1.0",
            notes=""
        )
        candidates.append(entry)
        
    return candidates

def main():
    test_texts, val_texts = load_hastika_texts()
    
    stats = {
        "multitarget-conan": {"raw": 0, "candidate": 0, "empty": 0, "unmapped": 0, "duplicates": 0, "conflicts": 0, "one_to_many": 0, "rejected": 0, "filtered_candidate": 0},
        "conan": {"raw": 0, "candidate": 0, "empty": 0, "unmapped": 0, "duplicates": 0, "conflicts": 0, "one_to_many": 0, "rejected": 0, "filtered_candidate": 0},
        "total": {"raw": 0, "candidate": 0, "empty": 0, "unmapped": 0, "duplicates": 0, "conflicts": 0, "one_to_many": 0, "rejected": 0, "filtered_candidate": 0}
    }
    
    rejection_reasons = {}
    
    candidates = []
    
    candidates.extend(process_mt_conan(stats))
    candidates.extend(process_conan(stats))
    
    with open(os.path.join(DATA_DIR, "candidates.jsonl"), 'w', encoding='utf-8') as f:
        for i, c in enumerate(candidates):
            c.kb_id = f"KB-{i+1:05d}"
            f.write(json.dumps(c.to_dict()) + "\n")
            stats[c.source]["candidate"] += 1
            stats["total"]["candidate"] += 1
            
    filtered_candidates = []
    rejected = []
    conflicts_unmapped = []
    
    # 1. Dedup & Conflict logic based on pairs (ht, cs)
    # 2. One-to-many detection (ht) -> multiple cs
    
    seen_pairs = {} # (ht_lower, cs_lower) -> list of entries
    seen_hts = {}   # ht_lower -> list of (ht_lower, cs_lower) pair keys
    
    for c in candidates:
        ht_lower = c.hate_text.lower()
        cs_lower = c.counter_speech.lower()
        pair_key = (ht_lower, cs_lower)
        
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = []
        seen_pairs[pair_key].append(c)
        
        if ht_lower not in seen_hts:
            seen_hts[ht_lower] = set()
        seen_hts[ht_lower].add(pair_key)
        
    # Process pairs for duplicates vs true conflicts
    processed_dedup = []
    
    for pair_key, entries in seen_pairs.items():
        if len(entries) == 1:
            processed_dedup.append(entries[0])
        else:
            # We have exact duplicate (HT, CS) pairs.
            # Check for metadata contradictions (Category, Target, Language, Source).
            # If all are essentially compatible (or we just pick the first), it's a dedup.
            # If they have contradictory categories, it's a true conflict.
            base = entries[0]
            is_conflict = False
            for e in entries[1:]:
                if e.category != base.category or e.target != base.target or e.language != base.language:
                    is_conflict = True
                    break
            
            if is_conflict:
                for e in entries:
                    e.quality_status = "conflict"
                    e.notes = "True conflict: Contradictory metadata for identical HT and CS."
                    stats[e.source]["conflicts"] += 1
                    stats["total"]["conflicts"] += 1
                    conflicts_unmapped.append(e)
            else:
                # Same HT, same CS, consistent metadata -> exact duplicate
                processed_dedup.append(base)
                # Count the remaining as duplicates
                for e in entries[1:]:
                    stats[e.source]["duplicates"] += 1
                    stats["total"]["duplicates"] += 1

    # Count one-to-many
    for ht_lower, pair_keys in seen_hts.items():
        if len(pair_keys) > 1:
            # Found one-to-many relationship
            # Let's count how many extra valid CS we have for this HT
            # The one_to_many stat tracks the total number of records participating in a 1-N relationship
            # or we just increment for every record involved.
            for pk in pair_keys:
                if len(seen_pairs[pk]) > 0:
                    src = seen_pairs[pk][0].source
                    stats[src]["one_to_many"] += 1
                    stats["total"]["one_to_many"] += 1

    # Process remaining for unmapped, safety, leakage
    for c in processed_dedup:
        if c.quality_status == "conflict":
            continue
            
        if c.category == "UNMAPPED":
            stats[c.source]["unmapped"] += 1
            stats["total"]["unmapped"] += 1
            c.quality_status = "unmapped"
            c.notes = "Unmapped category"
            conflicts_unmapped.append(c)
            continue
            
        # Leakage check
        safe_leakage, leak_reason = leakage_filter(c.hate_text, test_texts, val_texts)
        if not safe_leakage:
            stats[c.source]["rejected"] += 1
            stats["total"]["rejected"] += 1
            c.quality_status = "rejected"
            c.notes = leak_reason
            rejected.append(c)
            rejection_reasons[leak_reason] = rejection_reasons.get(leak_reason, 0) + 1
            continue
            
        # Safety check
        safe_safety, safety_reason = safety_filter(c.counter_speech)
        if not safe_safety:
            stats[c.source]["rejected"] += 1
            stats["total"]["rejected"] += 1
            c.quality_status = "rejected"
            c.notes = f"Safety: {safety_reason}"
            rejected.append(c)
            rejection_reasons[safety_reason] = rejection_reasons.get(safety_reason, 0) + 1
            continue
            
        c.quality_status = "filtered_candidate"
        stats[c.source]["filtered_candidate"] += 1
        stats["total"]["filtered_candidate"] += 1
        filtered_candidates.append(c)
        
    with open(os.path.join(DATA_DIR, "filtered_candidates.jsonl"), 'w', encoding='utf-8') as f:
        for c in filtered_candidates:
            f.write(json.dumps(c.to_dict()) + "\n")
            
    with open(os.path.join(DATA_DIR, "rejected.jsonl"), 'w', encoding='utf-8') as f:
        for c in rejected:
            f.write(json.dumps(c.to_dict()) + "\n")
            
    with open(os.path.join(DATA_DIR, "conflicts_unmapped.jsonl"), 'w', encoding='utf-8') as f:
        for c in conflicts_unmapped:
            f.write(json.dumps(c.to_dict()) + "\n")
            
    audit_report = {
        "stats": stats,
        "rejection_reasons": rejection_reasons
    }
    
    with open(os.path.join("reports", "m5_2_audit.json"), 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)
        
    with open(os.path.join("docs", "M5.2_COMPLETION_REPORT.md"), 'w', encoding='utf-8') as f:
        f.write("# M5.2 Candidate Mapping & Filtering Report (Corrected)\n\n")
        f.write("## Overview\n")
        f.write("Immutable sources were mapped to the 18-field KB schema, filtered, and deduplicated.\n")
        f.write("A distinction is explicitly maintained between **one-to-many counter-speech relationships** (same hate text, different valid counter-narratives) and **true annotation conflicts** (same text pair, contradictory metadata).\n\n")
        f.write("## Accounting\n")
        f.write(f"Total Raw: {stats['total']['raw']}\n")
        f.write(f"Total Candidate: {stats['total']['candidate']}\n")
        f.write(f"Empty/Invalid: {stats['total']['empty']}\n")
        f.write(f"Unmapped: {stats['total']['unmapped']}\n")
        f.write(f"Exact Duplicates (Removed): {stats['total']['duplicates']}\n")
        f.write(f"Valid One-to-Many Records Preserved: {stats['total']['one_to_many']}\n")
        f.write(f"True Conflicts (Contradictory Metadata): {stats['total']['conflicts']}\n")
        f.write(f"Rejected: {stats['total']['rejected']}\n")
        f.write(f"Final Filtered Candidates: {stats['total']['filtered_candidate']}\n\n")
        
        f.write("## Rejection Reasons\n")
        for reason, count in rejection_reasons.items():
            f.write(f"- {reason}: {count}\n")
            
    with open(os.path.join("docs", "m5-category-mapping.md"), 'w', encoding='utf-8') as f:
        f.write("# Category Mapping\n\n")
        f.write("## Multitarget-CONAN\n")
        f.write("- WOMEN -> Gender\n- LGBT+ -> Gender\n- JEWS -> Religion\n- MUSLIMS -> Religion\n- MIGRANTS -> Geo-political\n- POC -> Others\n- OTHER -> Others\n\n")
        f.write("## CONAN\n")
        f.write("- Islamophobia -> Religion\n")

if __name__ == "__main__":
    main()
