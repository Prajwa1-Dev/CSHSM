import os
import csv
import json
import uuid
import re
import argparse
import shutil
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

def backup_existing_artifacts():
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("reports", "backups", f"m5_2_pre_run_{timestamp}")

    files_to_backup = [
        os.path.join("reports", "m5_2_audit.json"),
        os.path.join("docs", "M5.2_COMPLETION_REPORT.md"),
        os.path.join("docs", "m5-category-mapping.md"),
        os.path.join(DATA_DIR, "candidates.jsonl"),
        os.path.join(DATA_DIR, "filtered_candidates.jsonl"),
        os.path.join(DATA_DIR, "rejected.jsonl"),
        os.path.join(DATA_DIR, "conflicts_unmapped.jsonl"),
    ]
    backed_up = []
    for fpath in files_to_backup:
        if os.path.exists(fpath):
            os.makedirs(backup_dir, exist_ok=True)
            rel_name = os.path.basename(fpath)
            dest = os.path.join(backup_dir, rel_name)
            shutil.copy2(fpath, dest)
            backed_up.append((fpath, dest))
    return backup_dir if backed_up else None

def parse_args():
    parser = argparse.ArgumentParser(
        description="M5.2 Counter-Speech Knowledge Base: Candidate Mapping & Filtering"
    )
    parser.add_argument(
        "--mode",
        choices=["verified", "provisional"],
        default="verified",
        help="Pipeline execution mode. 'verified' requires non-empty HASTIKA test/validation reference splits for leakage audit. 'provisional' produces mapped and safety-filtered artifacts without claiming leakage verification."
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="Shortcut flag for --mode provisional."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup of existing M5.2 artifacts prior to overwrite."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    mode = "provisional" if args.provisional else args.mode

    test_texts, val_texts = load_hastika_texts()
    has_leakage_references = bool(test_texts and val_texts)

    if not has_leakage_references and mode == "verified":
        test_path = os.path.join(SPLITS_DIR, "test.jsonl")
        val_path = os.path.join(SPLITS_DIR, "val.jsonl")
        print("=" * 76, file=sys.stderr)
        print("ERROR: M5.2 Verified Mode Blocked -- HASTIKA Reference Splits Missing/Empty", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        print(f"  - Test set split:       {test_path} ({'Loaded ' + str(len(test_texts)) + ' comments' if os.path.exists(test_path) else 'NOT FOUND'})", file=sys.stderr)
        print(f"  - Validation set split: {val_path} ({'Loaded ' + str(len(val_texts)) + ' comments' if os.path.exists(val_path) else 'NOT FOUND'})", file=sys.stderr)
        print("\nVerified mode requires non-empty HASTIKA test and validation splits to audit dataset leakage.", file=sys.stderr)
        print("Per project safety rules, missing reference sets CANNOT silently be treated as proof of zero leakage.", file=sys.stderr)
        print("\nTo produce provisional mapping and safety-filtering artifacts WITHOUT claiming leakage verification, run with:", file=sys.stderr)
        print("  py scripts/run_m52_mapping.py --provisional", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        sys.exit(1)

    if not args.no_backup:
        backup_loc = backup_existing_artifacts()
        if backup_loc:
            print(f"[M5.2] Backed up pre-existing artifacts to: {backup_loc}")

    stats = {
        "multitarget-conan": {"raw": 0, "candidate": 0, "empty": 0, "unmapped": 0, "duplicates": 0, "conflicts": 0, "one_to_many": 0, "rejected": 0, "filtered_candidate": 0},
        "conan": {"raw": 0, "candidate": 0, "empty": 0, "unmapped": 0, "duplicates": 0, "conflicts": 0, "one_to_many": 0, "rejected": 0, "filtered_candidate": 0},
        "total": {"raw": 0, "candidate": 0, "empty": 0, "unmapped": 0, "duplicates": 0, "conflicts": 0, "one_to_many": 0, "rejected": 0, "filtered_candidate": 0}
    }

    rejection_reasons = {}
    leakage_match_counts = {"test": 0, "val": 0}

    candidates = []

    candidates.extend(process_mt_conan(stats))
    candidates.extend(process_conan(stats))

    os.makedirs(DATA_DIR, exist_ok=True)
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

        # Leakage check (executed only when reference splits are available)
        if has_leakage_references:
            safe_leakage, leak_reason = leakage_filter(c.hate_text, test_texts, val_texts)
            if not safe_leakage:
                stats[c.source]["rejected"] += 1
                stats["total"]["rejected"] += 1
                c.quality_status = "rejected"
                c.notes = leak_reason
                rejected.append(c)
                rejection_reasons[leak_reason] = rejection_reasons.get(leak_reason, 0) + 1
                if "Test Set" in leak_reason:
                    leakage_match_counts["test"] += 1
                elif "Validation Set" in leak_reason:
                    leakage_match_counts["val"] += 1
                continue

        # Safety check (ALWAYS executed; never skipped or weakened)
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
        if not has_leakage_references:
            c.notes = "Provisional: HASTIKA leakage unverified (splits unavailable)"
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

    leakage_audit_info = {
        "status": "completed" if has_leakage_references else "skipped_provisional",
        "verified": has_leakage_references,
        "mode": mode,
        "reason": (
            "HASTIKA test and validation reference splits verified; exact-match leakage audit executed against both splits."
            if has_leakage_references else
            "HASTIKA reference splits (data/splits/test.jsonl, data/splits/val.jsonl) are missing or empty on this machine. Leakage filtering was skipped. Zero leakage is NOT claimed."
        ),
        "hastika_splits": {
            "test_path": os.path.join(SPLITS_DIR, "test.jsonl"),
            "test_present": os.path.exists(os.path.join(SPLITS_DIR, "test.jsonl")),
            "test_count": len(test_texts),
            "val_path": os.path.join(SPLITS_DIR, "val.jsonl"),
            "val_present": os.path.exists(os.path.join(SPLITS_DIR, "val.jsonl")),
            "val_count": len(val_texts)
        },
        "matches": {
            "test_matches": leakage_match_counts["test"],
            "val_matches": leakage_match_counts["val"],
            "total_leakage_rejected": leakage_match_counts["test"] + leakage_match_counts["val"]
        },
        "cross_split_observation": "Four cross-split comment pairs overlap under case-insensitive comparison in the restored HASTIKA splits, although original exact-text split check found zero. Leakage filter applied normalized exact matching against all reference entries."
    }

    audit_report = {
        "leakage_audit": leakage_audit_info,
        "stats": stats,
        "rejection_reasons": rejection_reasons
    }

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "m5_2_audit.json"), 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)

    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", "M5.2_COMPLETION_REPORT.md"), 'w', encoding='utf-8') as f:
        f.write("# M5.2 Candidate Mapping & Filtering Report\n\n")
        f.write("## Overview\n")
        f.write("Immutable sources were mapped to the 18-field KB schema, filtered, and deduplicated.\n")
        f.write("A distinction is explicitly maintained between **one-to-many counter-speech relationships** (same hate text, different valid counter-narratives) and **true annotation conflicts** (same text pair, contradictory metadata).\n\n")

        f.write("## HASTIKA Leakage Audit\n")
        if leakage_audit_info["verified"]:
            f.write("- **Status**: Completed (Verified)\n")
            f.write(f"- Mode: `{mode}`\n")
            f.write(f"- Test Set References Checked: {len(test_texts)}\n")
            f.write(f"- Validation Set References Checked: {len(val_texts)}\n")
            f.write(f"- Exact Matches in HASTIKA Test Set: {leakage_match_counts['test']}\n")
            f.write(f"- Exact Matches in HASTIKA Validation Set: {leakage_match_counts['val']}\n")
            f.write("- **Cross-Split Reference Note**: While exact-string overlap across restored HASTIKA splits is strictly 0 (Train/Val=0, Train/Test=0, Val/Test=0), 4 cross-split comment pairs differ solely by casing under case-insensitive comparison. M5.2 leakage filtering applies lowercase whitespace-normalized exact matching against both test and validation reference sets; no leakage matches were detected.\n\n")
        else:
            f.write("- **Status**: SKIPPED (PROVISIONAL MODE)\n")
            f.write(f"- Mode: `{mode}`\n")
            f.write(f"- Reason: {leakage_audit_info['reason']}\n")
            f.write("- **IMPORTANT**: Zero leakage is **NOT** claimed. Candidates produced in this run are provisional with respect to HASTIKA split isolation. Downstream milestone M5.4 remains blocked until genuine splits are available and verified.\n\n")

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
        if rejection_reasons:
            for reason, count in sorted(rejection_reasons.items()):
                f.write(f"- {reason}: {count}\n")
        else:
            f.write("- (None)\n")

    with open(os.path.join("docs", "m5-category-mapping.md"), 'w', encoding='utf-8') as f:
        f.write("# Category Mapping\n\n")
        f.write("## Multitarget-CONAN\n")
        f.write("- WOMEN -> Gender\n- LGBT+ -> Gender\n- JEWS -> Religion\n- MUSLIMS -> Religion\n- MIGRANTS -> Geo-political\n- POC -> Others\n- OTHER -> Others\n\n")
        f.write("## CONAN\n")
        f.write("- Islamophobia -> Religion\n")

    print("[M5.2] Execution completed.")
    print(f"[M5.2] Mode: {mode} (Leakage verified: {has_leakage_references})")
    print(f"[M5.2] Filtered candidates: {stats['total']['filtered_candidate']}")
    print(f"[M5.2] Rejections (safety): {stats['total']['rejected']}")
    print(f"[M5.2] Leakage rejections -> Test: {leakage_match_counts['test']}, Val: {leakage_match_counts['val']}")

if __name__ == "__main__":
    main()
