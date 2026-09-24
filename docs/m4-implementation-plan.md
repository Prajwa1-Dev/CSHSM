# M4 Implementation Plan: XLM-R Hate Detection & Classification

## 1. Objective
The objective of Milestone 4 is to build, train, and evaluate the XLM-R-based hate-speech classification component of the CSHSM project. This component acts as the primary gatekeeper in the architecture. It must accurately perform two tasks:
1. Detect whether a comment is `Hate` or `Non-Hate`.
2. For `Hate` comments, classify them into one of six specific categories: `Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, or `Others`.

## 2. Existing Data
- **Final Usable Records**: 7,995
- **Splits**: Train (6,394), Validation (796), Test (805).
- **Label Distribution**: Highly balanced (Hate 49%, Non-Hate 51%).
- **Category Distribution (Hate only)**: Highly imbalanced (Gender dominates, Geo-political and Violence are sparse).
- **Format**: Cleaned, deduplicated JSONL files with code-mixing, Kannada script, and emojis preserved.

## 3. Model Architecture Options
**Option A: Two Separate Models (Cascade)**
- *Design*: One XLM-R model trained solely for Hate/Non-Hate classification, and a second XLM-R model trained solely on Hate data for Category classification.
- *Trade-offs*: Easier to tune loss and hyperparameters independently. However, it requires storing and loading two massive Transformer backbones, doubling the RAM, disk space, and inference time requirements.

**Option B: Single Multi-Task Model (Shared Backbone)**
- *Design*: A single XLM-R base model with two task-specific linear heads:
  - Head 1: Binary classification (Hate vs. Non-Hate).
  - Head 2: 6-Class classification (Hate Categories).
- *Trade-offs*: Halves the memory/disk footprint. The shared representation can act as a regularizer, potentially improving both tasks (as hate-category features might aid general hate detection). Requires custom loss masking because the category head should only incur loss on true `Hate` samples.

## 4. Recommended Architecture
Given the strict disk space constraint (~7.75 GB limit) and typical local memory constraints, **Option B (Multi-Task XLM-R)** is strongly recommended. 
- It requires only one backbone to be stored and loaded in RAM.
- A single forward pass yields both outputs, minimizing inference latency for the downstream RAG pipeline.
- The total loss will be computed as: `Loss = L_binary + (L_category * mask)`, where `mask = 1` if the true label is Hate, else `0`.

## 5. XLM-R Variant Recommendation
- **Recommended Variant**: `xlm-roberta-base` (278M parameters).
- **Why**: It possesses excellent multilingual capabilities, naturally supporting both English and Kannada tokenization (via SentencePiece). It natively handles code-mixing without requiring language-specific preprocessing.
- **Constraints**: The `xlm-roberta-large` (550M parameters) variant is too large for the current disk constraints (requires >2.2GB per checkpoint) and is slower to fine-tune. The `base` model strikes the perfect balance for research prototyping.

## 6. Tokenization Strategy
- **Tokenizer**: Standard `XLMRobertaTokenizerFast`.
- **Input Field**: The normalized `Comment` field.
- **Label Encodings**:
  - `Label`: `Non-Hate` -> 0, `Hate` -> 1.
  - `Hate Category`: Mapped 0-5. For `Non-Hate` records, a dummy value (e.g., `-1`) will be assigned and ignored during loss calculation via PyTorch's `ignore_index` or explicit masking.
- **Length**: Maximum sequence length of **128**. 
  - *Distribution Analysis*: We explicitly computed the XLM-R tokenizer (`xlm-roberta-base`) sequence lengths on the M3 `hastika_cleaned.jsonl` dataset. The true token-length distribution is:
    - 50th percentile: 18 tokens
    - 75th percentile: 30 tokens
    - 90th percentile: 51 tokens
    - 95th percentile: 70 tokens
    - 99th percentile: 125 tokens
    - Maximum: 366 tokens
  - *Recommendation*: Since 99% of all records fall under 128 tokens, a `max_length` of 128 optimally covers the dataset without wasting padding computation memory.
- **Padding/Truncation**: Pad to longest in batch; truncate sequences longer than max length.
- **Preservation**: Emojis, Kannada scripts, and `#hashtags` remain intact and will be tokenized using the model's native vocabulary.

## 7. Training Strategy
- **Optimizer**: AdamW (`lr=2e-5`).
- **Batch Size**: 16 (or 8 with gradient accumulation steps=2 if VRAM/RAM is limited).
- **Epochs**: 5 to 10.
- **Scheduler**: Linear learning rate decay with a 10% warmup.
- **Weight Decay**: 0.01.
- **Checkpoint Selection**: The best model will be selected based on the **Unweighted Average of Binary Macro-F1 and Category Macro-F1**. Both tasks are critical, and relying solely on the binary metric ignores the model's performance on the 6-class category task.
- **Early Stopping**: Monitor the unweighted average Macro-F1 with a patience of 3 epochs.
- **Reproducibility**: Random seed fixed to `42` globally (NumPy, PyTorch, Transformers).

## 8. Loss Tracking & Imbalance Strategy
- **Binary Task**: No imbalance treatment is needed (49% vs 51% is balanced).
- **Category Task**: Severe imbalance requires intervention. We will apply **Weighted Cross-Entropy Loss** for the category head using inverse class frequencies derived exclusively from the training split.
- **Loss Formulation**:
  `Total Loss = Binary Loss + λ × Masked Category Loss`
  - `λ` is a configurable scaling factor.
  - The Category Loss is *only* applied to true `Hate` training examples (loss is masked to 0 for Non-Hate records).
- **Tracking**: Binary loss, Masked Category loss, and Total loss will be explicitly calculated and tracked separately in the logs.

## 9. Evaluation Strategy
- **Binary Hate Detection (Head 1)**:
  - Metrics: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix.
- **Hate Category Classification (Head 2)**:
  - We will report two distinct category evaluations:
    1. **Oracle Category Evaluation**: Evaluated exclusively on true `Hate` examples passed to the category head. Measures the pure performance of the category classifier.
    2. **Cascaded Category Evaluation**: Evaluated by using the actual predicted Hate/Non-Hate output from the binary head. This measures real-world end-to-end pipeline accuracy (incorporating false negatives from the binary head).
  - Metrics: Per-class Precision, Recall, F1, Macro-F1, Confusion Matrix.

## 10. Baselines
We will implement two non-deep-learning baselines to justify the XLM-R model:
1. **Binary Baseline**: Always predict the majority binary class (`Non-Hate`).
2. **Category Baseline**: Among true `Hate` examples, always predict the majority Hate category (`Gender`). Category prediction will not be evaluated for Non-Hate examples.

## 11. Error Analysis
We will perform fine-grained error analysis, logging confident false predictions to `reports/m4_error_analysis.md`.
- **Category-specific analysis**: We will explicitly analyze errors within Gender, Political, Religion, Geo-political, Violence, and Others.
- **Text Characteristics analysis**: We will evaluate performance disparities across:
  - Kannada-heavy comments
  - English-heavy comments
  - Code-mixed comments
  - Short vs. Long comments
  - Emoji-containing comments

## 12. Resource Requirements
- **Disk Space**: ~1.1 GB for the base model.
- **Checkpoints**: To respect the 7.75 GB disk limit, we will save *only the single best checkpoint* based on validation macro-F1, discarding intermediate epoch weights.
- **RAM/VRAM**: Fine-tuning XLM-R base at batch size 8-16 requires roughly 6-8 GB of memory. 

## 13. Experiment Tracking
All experiments will log locally to `experiments/exp_<timestamp>.json`. Tracking will include:
- Hyperparameters (LR, batch size, weight decay, epochs).
- Data version (implicitly tied to the M3 split).
- Epoch-wise train/val loss.
- Final test metrics (Binary F1, Category Macro-F1).
This strict logging ensures complete reproducibility.

## 14. Testing Strategy
- **`test_dataset.py`**: Verify tokenization, padding, and that the custom PyTorch `Dataset` correctly yields input IDs, attention masks, and both sets of labels. Verify label masking (e.g., `-1`) works for Non-Hate categories.
- **`test_model.py`**: Instantiate the custom multi-task model with dummy weights and perform a forward pass to assert output tensor dimensions match `(batch, 2)` and `(batch, 6)`.
- **`test_loss.py`**: Assert that the custom masked loss function correctly calculates gradients only for valid category labels.

## 15. Expected Outputs
Once implemented, M4 will create:
- `src/models/multitask_xlmr.py` (Model definition)
- `src/models/trainer.py` (Training loop)
- `models/cshsm-xlmr-best/` (Final trained weights)
- `experiments/exp_*.json` (Logs)
- `reports/m4_evaluation.md` (Metrics, Baselines, Confusion Matrices, Error Analysis)

## 16. Risks and Mitigations
- **OOM (Out of Memory) Errors**: Mitigated by reducing batch size to 8 and using gradient accumulation steps.
- **Disk Full**: Mitigated by overwriting checkpoints and retaining only the single best model state.
- **Head Domination**: Mitigated by tracking individual head losses. If the binary head converges too fast, we can apply dynamic loss weighting.
- **Test Set Leakage**: Mitigated by loading the test split *only* at the very end of the script for final evaluation. Hyperparameters will be tuned strictly on the validation split.
