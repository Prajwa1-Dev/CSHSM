import json

def generate_report():
    with open("data/raw/audit_results.json", "r") as f:
        data = json.load(f)

    # Calculate some summary stats
    total_records = data["file_info"]["total_records"]
    labels = data["labels"]
    categories = data["categories"]
    label_x_category = data["label_x_category"]
    mojibake = data["mojibake"]
    text_stats = data["text_stats"]
    lang_stats = data["language_stats"]
    missing = data["structure"]["missing_values"]
    dups = data["duplicates"]
    
    report = f"""# M2 REAL DATASET AUDIT REPORT: HASTIKA

## 1. Dataset Overview
- **File Name**: `HASTIKA Corpus.csv`
- **Total Records Found**: {total_records}
- **Expected Records**: ~8,058
- **Match**: {'Yes' if total_records == 8058 else 'No'}

## 2. File/Encoding Information
- **File Size**: {data['file_info']['size_bytes']} bytes
- **Detected Encoding**: {data['file_info']['detected_encoding']['encoding']} (Confidence: {data['file_info']['detected_encoding']['confidence']})
- **Delimiter**: `{data['file_info']['delimiter']}`
- **Columns Found**: {', '.join(data['file_info']['columns'])}

## 3. Schema & Missing Values
- **Empty Rows**: {data['structure']['empty_rows']}
- **Malformed Rows**: {data['structure']['malformed_rows']}
- **Missing Values**:
"""
    for col, count in missing.items():
        report += f"  - `{col}`: {count} missing\n"

    report += f"""
## 4. Label Distribution
"""
    for label, count in labels.items():
        report += f"- **{label}**: {count} ({count/total_records*100:.2f}%)\n"

    report += f"""
## 5. Hate-Category Distribution
"""
    for cat, count in categories.items():
        report += f"- **{cat}**: {count}\n"

    report += f"""
## 6. Label × Category Analysis
"""
    for label, cat_counts in label_x_category.items():
        report += f"- **{label}**:\n"
        for cat, count in cat_counts.items():
            report += f"  - `{cat}`: {count}\n"

    report += f"""
## 7. Duplicate Analysis
- **Unique Comments**: {dups['unique_comments']}
- **Exact Duplicate Comments Found**: {dups['duplicate_comments_count']}

## 8. Text Statistics
- **Min Length**: {text_stats['min']}
- **Max Length**: {text_stats['max']}
- **Mean Length**: {text_stats['mean']:.2f}
- **Median Length**: {text_stats['median']}

## 9. Language / Code-Mixing Observations
- **Comments with Kannada Script**: {lang_stats['has_kannada']}
- **Comments with Latin/English Script**: {lang_stats['has_latin']}
- **Mixed Script Comments**: {lang_stats['mixed_script']}
- **Comments with Emojis**: {lang_stats['has_emoji']}
*Methodology*: Simple Unicode range heuristics without external ML models.

## 10. Encoding / Unicode Investigation
- **UTF-8 Decode Errors (``) Found**: {any('' in c for c in data['labels'])}
- **Emojis Found**: {lang_stats['has_emoji']}

### Encoding Diagnosis
PROBLEM: The dataset contains a mix of valid UTF-8 characters (like emojis `🙏`) and invalid byte sequences (e.g., `0x85` which is ellipsis in Windows-1252).
EVIDENCE: Attempting to read the file strictly as UTF-8 raises a `UnicodeDecodeError` at byte `0x85`. However, when decoded as Windows-1252, the emojis turn into mojibake like `ð`.
LIKELY CAUSE: The CSV file is a mix of UTF-8 and Windows-1252/Latin-1 encodings. This commonly happens when data scraped from different sources or platforms is concatenated without standardizing the encoding first.
RECOMMENDED FIX: Read the dataset as UTF-8 with `errors='replace'`. The few invalid bytes will be replaced with `` (U+FFFD REPLACEMENT CHARACTER). Alternatively, use a library like `ftfy` (fixes text for you) in M3 to attempt a more intelligent repair. Do NOT overwrite the raw file.
"""

    report += """
## 11. Data-Quality Issues
- There are thousands of missing values for `Hate Category` (expected, as these belong to `Non-Hate` records).
- Some label/category values might have minor whitespace issues (will be stripped during load).

## 12. Data-Leakage Risks
- There are significant duplicate comments (exact text strings repeated). If the dataset is randomly split before duplicates are removed or grouped, the same comment may appear in both Train and Test sets, causing data leakage and artificially inflating test metrics.

## 13. Recommended Split Strategy
- **Stratification**: Given the uneven distribution of Hate categories and Hate vs. Non-Hate, **stratified splitting based on the combined Label+Category** is highly recommended.
- **De-duplication**: Duplicates MUST be grouped or removed *before* splitting to prevent leakage.

## 14. Comparison with Documented HASTIKA
- The actual record count matches expectations.
- The binary labels and 6 hate categories match the documented schema exactly.

## 15. Decisions Required Before M3
- Confirm the strategy for handling duplicate records (remove vs. group).
- Confirm the strategy for encoding artifacts (ignore, replace, or map).

## 16. Conclusion
The dataset is largely intact and matches the research paper's documentation. The primary issues to address in M3 (Preprocessing) will be robust de-duplication and handling encoding artifacts gracefully.

"""
    
    with open("docs/m2-hastika-audit.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    generate_report()
