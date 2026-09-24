# M2 Dataset Pipeline Documentation

## 1. Purpose of M2
The objective of Milestone 2 is to prepare a robust, reproducible data processing pipeline (loading, validating, cleaning, analyzing, and splitting) specifically tailored for the HASTIKA dataset, *without* requiring the actual dataset file to be present during setup.

## 2. HASTIKA's Role in CSHSM
HASTIKA is the primary dataset used for the CSHSM project. It provides the foundational hate-speech and counter-speech records needed for classification (XLM-R), vector retrieval (BGE-M3 + FAISS), and generation validation.

## 3. Expected Dataset Structure
Based on the project plan, the dataset is expected to contain:
- A text comment field.
- A hate/non-hate binary label.
- A hate-category label (for hate examples).
*Note: The exact column names are currently configurable pending the receipt of the actual dataset.*

## 4. Data Directory Structure
The data is isolated from the codebase to ensure raw files are never modified and splits are reproducible:
- `data/raw/`: Immutable source files.
- `data/processed/`: Cleaned and standardized files.
- `data/splits/`: Deterministic train/val/test splits.
- `data/knowledge_base/`: Curated trusted responses for M5.

## 5. Loading Process
Implemented in `src.data.loader.DatasetLoader`. It expects `.json` or `.jsonl` files and throws explicit errors for missing files or unsupported formats to prevent silent failures.

## 6. Validation Process
Implemented in `src.data.validator.DatasetValidator`. It checks that the configured `text_field` and `label_field` exist in every record. 

## 7. Cleaning Process
Implemented in `src.data.cleaner.TextCleaner`. It performs conservative preprocessing:
- Unicode normalization (NFC)
- Whitespace reduction (collapsing tabs/newlines)
- Preservation of emojis and semantic context.

## 8. Analysis Process
Implemented in `src.data.analyzer.DatasetAnalyzer`. Once HASTIKA is available, it calculates:
- Total records and missing values
- Duplicate comments
- Label distributions
- Text length statistics (min, max, mean)
*Currently, the analyzer gracefully reports that execution is pending the real dataset.*

## 9. Split Strategy
Implemented in `src.data.splitter.DatasetSplitter`. Current implementation performs deterministic random splitting. True stratified splitting will be evaluated and implemented after the real HASTIKA dataset schema and label distribution are inspected. The exact default ratio is 80/10/10 but remains configurable.

## 10. Reproducibility
- All random operations (e.g., shuffling) use a fixed seed (default 42).
- Cleaned datasets and splits are saved explicitly to disk.
- Raw data is strictly read-only.

## 11. Current Blocker
The actual HASTIKA dataset file has not yet been received. Therefore, the pipeline cannot be executed end-to-end against real data.

## 12. Next Steps Once Dataset is Provided
1. Place the HASTIKA file in `data/raw/`.
2. Update `m2_config.ini` with the exact file path and confirmed schema column names.
3. Run the pipeline script to produce `data/splits/`.
4. Run the analyzer to generate the official dataset statistics.
