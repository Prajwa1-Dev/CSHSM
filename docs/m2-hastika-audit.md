# M2 REAL DATASET AUDIT REPORT: HASTIKA

## 1. Dataset Overview
- **File Name**: `HASTIKA Corpus.csv`
- **Total Records Found**: 8058
- **Expected Records**: ~8,058
- **Match**: Yes

## 2. File/Encoding Information
- **File Size**: 769805 bytes
- **Detected Encoding**: cp775 (Confidence: 0.24504268225017253)
- **Delimiter**: `,`
- **Columns Found**: id, Comment, Label, Hate Category

## 3. Schema & Missing Values
- **Empty Rows**: 0
- **Malformed Rows**: 0
- **Missing Values**:
  - `Hate Category`: 4108 missing

## 4. Label Distribution
- **Non-Hate**: 4108 (50.98%)
- **Hate**: 3950 (49.02%)

## 5. Hate-Category Distribution
- **Others**: 561
- **Political**: 699
- **Gender**: 1703
- **Geo-political**: 232
- **Religion**: 478
- **Violence**: 277

## 6. Label × Category Analysis
- **Non-Hate**:
  - ``: 4108
- **Hate**:
  - `Others`: 561
  - `Political`: 699
  - `Gender`: 1703
  - `Geo-political`: 232
  - `Religion`: 478
  - `Violence`: 277

## 7. Duplicate Analysis
- **Unique Comments**: 8004
- **Exact Duplicate Comments Found**: 48

## 8. Text Statistics
- **Min Length**: 5
- **Max Length**: 1084
- **Mean Length**: 75.19
- **Median Length**: 52.0

## 9. Language / Code-Mixing Observations
- **Comments with Kannada Script**: 56
- **Comments with Latin/English Script**: 8055
- **Mixed Script Comments**: 56
- **Comments with Emojis**: 1119
*Methodology*: Simple Unicode range heuristics without external ML models.

## 10. Encoding / Unicode Investigation
- **UTF-8 Decode Errors (``) Found**: True
- **Emojis Found**: 1119

### Encoding Diagnosis
PROBLEM: The dataset contains a mix of valid UTF-8 characters (like emojis `🙏`) and invalid byte sequences (e.g., `0x85` which is ellipsis in Windows-1252).
EVIDENCE: Attempting to read the file strictly as UTF-8 raises a `UnicodeDecodeError` at byte `0x85`. However, when decoded as Windows-1252, the emojis turn into mojibake like `ð`.
LIKELY CAUSE: The CSV file is a mix of UTF-8 and Windows-1252/Latin-1 encodings. This commonly happens when data scraped from different sources or platforms is concatenated without standardizing the encoding first.
RECOMMENDED FIX: Read the dataset as UTF-8 with `errors='replace'`. The few invalid bytes will be replaced with `` (U+FFFD REPLACEMENT CHARACTER). Alternatively, use a library like `ftfy` (fixes text for you) in M3 to attempt a more intelligent repair. Do NOT overwrite the raw file.

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

