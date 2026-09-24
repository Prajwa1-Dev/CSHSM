# M3 Implementation Plan: Dataset Preprocessing

## 1. Raw-Data Preservation
- **Why**: To maintain data integrity and reproducibility.
- **What**: The original `data/raw/HASTIKA Corpus.csv` will be strictly read-only. All modifications happen in memory, and the final outputs will be written to `data/processed/` and `data/splits/`.
- **Loss**: None.
- **Verification**: Ensure the file modification timestamp and hash of `HASTIKA Corpus.csv` remain unchanged.

## 2. Encoding Diagnosis and Safe Repair
- **Why**: The file is a mix of valid UTF-8 (including emojis) and invalid byte sequences. Blindly replacing errors or hardcoding specific mappings destroys potentially recoverable characters and masks the root cause.
- **What**: First, diagnose the actual mixed-encoding pattern in the raw file by isolating and identifying all undecodable byte sequences. Affected records and exact byte sequences will be explicitly logged. Replacement with `U+FFFD` will *only* be used when the data is determined to be genuinely unrecoverable, and the decision is formally documented. No blind hard-coding of mappings (like `0x85` to `…`) will occur prior to diagnosis.
- **Loss**: Only genuinely unrecoverable bytes will be replaced.
- **Verification**: A detailed log of all encoding anomalies and repairs/warnings will be generated in `preprocessing_report.json`.

## 3. Unicode Normalization
- **Why**: To ensure text consistency (e.g., combining accents vs. precomposed characters).
- **What**: Apply Unicode NFC (Normalization Form C). NFC is safest for Indic scripts (like Kannada) as it composes characters correctly for rendering and tokenization without breaking conjuncts.
- **Loss**: Visually identical but byte-different representations are unified.
- **Verification**: Unit tests verifying that sample Kannada conjuncts and English accents are normalized properly without visual or semantic corruption.

## 4. Whitespace Normalization
- **Why**: Social media comments contain erratic spacing, tabs, and line breaks that add noise to transformers.
- **What**: Strip leading/trailing whitespace, replace `\n`, `\r`, `\t` with spaces, and collapse multiple spaces into a single space.
- **Loss**: Original whitespace formatting is lost.
- **Verification**: Unit tests ensuring string length is reduced but all words remain correctly separated.

## 5. Label Normalization
- **Why**: To prevent duplicate classes due to typos or casing.
- **What**: Strip whitespace and capitalize the first letter to strictly enforce `Hate` and `Non-Hate` binaries.
- **Loss**: Minor formatting noise.
- **Verification**: Assert that the unique labels in the cleaned dataset are exactly `["Hate", "Non-Hate"]`, and record the count of label normalizations in `preprocessing_report.json`.

## 6. Hate-Category Normalization
- **Why**: To ensure consistent classes for the multiclass head.
- **What**: Strip whitespace and format to standard casing (`Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`). Force empty strings for `Non-Hate` records.
- **Loss**: None.
- **Verification**: Assert that the unique categories for `Hate` match the exact 6 expected types, and record the count of category normalizations in `preprocessing_report.json`.

## 7. Duplicate Handling & Conflict Resolution
- **Why**: Duplicates distort dataset statistics and cause train-test leakage. Conflicting annotations poison the classifier.
- **What**: 
  - Exact duplicate comments with *identical* labels and categories will be deduplicated into a single record.
  - Identical comments with *conflicting* annotations (different labels/categories) will be flagged as "annotation conflicts". They will NOT be automatically deleted, but rather excluded from the initial training/evaluation splits and kept aside. The decision and count will be documented.
- **Loss**: Redundant identical rows are reduced. Conflicting rows are excluded from training data.
- **Verification**: `preprocessing_report.json` will accurately record the number of duplicates deduplicated and the number of annotation conflicts flagged.

## 8. Emoji Preservation
- **Why**: Emojis carry vital sentiment and context in social media hate speech.
- **What**: Emojis will be completely preserved. 
- **Loss**: None.
- **Verification**: Unit tests ensuring strings with complex emojis pass through unaltered.

## 9. Kannada-Script Preservation
- **Why**: The project targets Kannada code-mixed text.
- **What**: Kannada characters will be preserved through the pipeline.
- **Loss**: None.
- **Verification**: Unit tests ensuring semantic and character preservation of Kannada text *after* NFC and whitespace normalization (rather than strict byte-for-byte identity, which would fail after NFC).

## 10. English/Code-Mixed Text Preservation
- **Why**: Code-mixing is central to the HASTIKA dataset.
- **What**: Mixed English-Latin and Kannada scripts are preserved natively.
- **Loss**: None.
- **Verification**: Visual inspection and testing of mixed-script samples.

## 11. URLs, Mentions, and Hashtags
- **Why**: URLs and direct usernames add noise and risk privacy. Hashtags often contain semantic meaning.
- **What**: 
  - URLs will be masked with `[URL]`.
  - `@usernames` will be masked with `[USER]`.
  - Hashtags (`#word`) will be strictly preserved.
- **Loss**: Specific links and username targets are generalized.
- **Verification**: Record the total counts of URLs and mentions replaced in `preprocessing_report.json`.

## 12. Empty/Invalid Records
- **Why**: Null text cannot be processed.
- **What**: Records missing both label and text, or where text reduces to empty after cleaning, will be removed.
- **Loss**: Useless/empty rows.
- **Verification**: Record empty records removed in `preprocessing_report.json`.

## 13. Data Leakage Prevention
- **Why**: Testing on training data yields artificially high accuracy.
- **What**: Deduplication (Step 7) occurs globally *before* any splitting. The split is performed solely on unique comments (excluding flagged conflicts).
- **Loss**: None.
- **Verification**: Script assertion confirming the intersection of train, val, and test texts is strictly empty.

## 14. Train/Validation/Test Split Strategy
- **Why**: Standard ML evaluation requires disjoint sets.
- **What**: 80% Train, 10% Validation, 10% Test.
- **Loss**: None.
- **Verification**: Count output sizes against the processed universe of records.

## 15. Label/Category Stratification
- **Why**: The dataset is imbalanced (e.g., Geo-political has only 232 records).
- **What**: First, verify that every composite class (`Label_Category`) has sufficient sample counts to support an 80/10/10 split without breaking. Once verified, perform stratified splitting using the composite key.
- **Loss**: None.
- **Verification**: Calculate and log the class distributions across train, val, and test to ensure robust representation.

## 16. Random Seed and Reproducibility
- **Why**: Scientific integrity requires repeatable results.
- **What**: Use a fixed random seed (e.g., 42 from configuration) for the stratifier.
- **Loss**: None.
- **Verification**: Re-running the split yields identically distributed output files.

## 17. Output Directory Structure
- **Why**: Keep outputs organized and isolated.
- **What**: 
  - `data/processed/hastika_cleaned.jsonl`
  - `data/processed/annotation_conflicts.jsonl`
  - `data/processed/preprocessing_report.json`
  - `data/splits/train.jsonl`
  - `data/splits/val.jsonl`
  - `data/splits/test.jsonl`
- **Loss**: None.

## 18. Preprocessing Report (`preprocessing_report.json`)
- **Why**: To provide complete transparency over data alterations.
- **What**: Will strictly record:
  - raw record count
  - processed record count
  - duplicates removed
  - conflicts flagged
  - empty records removed
  - URLs replaced
  - mentions replaced
  - encoding repairs/warnings
  - label normalizations
  - category normalizations

## 19. Tests That Should Be Added
- **Why**: To prevent regressions.
- **What**: 
  - `test_kannada_semantic_preservation()`
  - `test_url_mention_masking()`
  - `test_conflict_flagging()`
  - `test_composite_stratification_counts()`
- **Loss**: None.
- **Verification**: `pytest` execution.

## 20. M3 Documentation
- **Why**: Transparency and record keeping.
- **What**: Generate `M3_COMPLETION_REPORT.md` documenting pipeline execution against the actual data.
- **Loss**: None.

## 21. Rollback/Reproducibility Strategy
- **Why**: Data pipelines often need tweaking. Manual edits ruin this.
- **What**: The entire M3 process will be encapsulated in a single reproducible command. Deleting the `data/processed` and `data/splits` folders completely reverts the system to the M2 state. 
