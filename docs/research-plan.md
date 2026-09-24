# Research Plan

This document outlines the critical research questions that must be answered before final implementation decisions are made for subsequent milestones.

## 1. Data
- **Hate-speech datasets**: Which dataset(s) are most appropriate for training/evaluating the XLM-R classifier? (e.g., HateXplain, Multilingual Hate Speech Dataset).
- **Counter-speech datasets**: Which dataset(s) provide high-quality, trusted responses for the FAISS vector database? (e.g., CONAN, Multi-TARGET, PROCON).
- **Language support**: What languages will the system officially support? (Affects dataset choice and model selection).
- **Labels/Categories**: What taxonomy of hate speech will the system recognize? (e.g., Racism, Sexism, Homophobia).
- **Licensing**: What are the licensing restrictions of the chosen datasets? (Must permit research use).
- **Preprocessing**: What specific text normalization steps are required? (e.g., handling URLs, mentions, emojis, casing).
- **Data Integrity**: Are there duplicate records or risks of data leakage between the retrieval database and the evaluation test set?

## 2. Classification
- **Model Suitability**: Is XLM-R the most appropriate model, or are there better alternatives for this specific task?
- **Formulation**: Should classification be a binary task (hate/non-hate) followed by a multiclass task (category), or a single multi-label task?
- **Baselines**: What is the appropriate baseline to compare the XLM-R classifier against?
- **Metrics**: What evaluation metrics are most appropriate? (e.g., Macro F1, Precision, Recall).

## 3. Knowledge Base Construction
- **Curation & Quality**: What criteria define a high-quality, trusted counter-speech response?
- **Safety**: How do we ensure no toxic or counter-hate responses are included in the curated base?
- **Coverage**: Does the curated Knowledge Base adequately cover the taxonomy of hate speech the system aims to handle?
- **Duplication**: How will duplicates and near-duplicates be handled to optimize the vector space?
- **Leakage Prevention**: How do we ensure strict separation between the Knowledge Base and the test sets used for evaluation?

## 4. Retrieval
- **Model Suitability**: Is BGE-M3 the optimal embedding model for capturing the semantic nuance of counter-speech?
- **Embedding Strategy**: Should we embed just the hate comment, the counter-speech response, or both? How does this affect similarity search?
- **FAISS Configuration**: What FAISS index type (e.g., Flat L2, IVFFlat) is most appropriate for the actual size of our knowledge base?
- **Hyperparameter (K)**: What value of K (Top-K results) should be retrieved?
- **Similarity Metric**: How should similarity be calculated? (e.g., Cosine similarity, L2 distance).
- **Threshold Calibration**: What statistical procedure will be used to calibrate the routing threshold $\tau$?

## 5. Generation
- **Model Feasibility**: Which LLM APIs are feasible regarding cost, latency, and rate limits?
- **Local vs. API**: Which models, if any, can realistically run locally within the 7.75 GB storage constraint?
- **Licensing/Cost**: What are the financial implications of the API choices for the fan-out?
- **Model Count**: What is the optimal number of generation models to include in the fan-out? (Starting assumption is 5).
- **Concurrency Evaluation**: How will the performance (latency, success rate) of concurrent API calls be evaluated?

## 6. Safety
- **Detection Methods**: How should unsafe, toxic, or escalatory counter-speech be detected automatically? (e.g., Perspective API, local regex, Llama-Guard).
- **Handling Escalation**: How do we define and detect when a generated response inadvertently amplifies hate or escalates the situation?
- **Total Failure**: What specific safe fallback message should be used when all candidates fail safety checks?

## 7. Evaluation
- **Classification Eval**: How will the XLM-R model's performance be quantified?
- **Retrieval Eval**: How will the FAISS retrieval quality be measured? (e.g., Recall@K, MRR).
- **Generation Eval**: How will generated counter-speech be evaluated without relying on automated n-gram metrics like BLEU/ROUGE? (e.g., LLM-as-a-judge).
- **System Eval**: How will the complete hybrid system's end-to-end performance (latency, fallback rate, success rate) be evaluated?
- **Human Eval**: How should a human evaluation study be designed to validate the LLM-as-a-judge scores?

## 8. Experiments and Ablations
The following experiments must be designed and executed to validate the architecture:
- **Retrieval-only baseline**: System performance when forced to use FAISS.
- **Generation-only baseline**: System performance when forced to use a single LLM.
- **Hybrid system**: The proposed architecture combining retrieval and generation.
- **Varying K**: Impact of retrieving different numbers of documents.
- **Varying $\tau$**: Impact of adjusting the routing threshold on system behavior.
- **Calibrated vs. Uncalibrated $\tau$**: Value added by empirical calibration.
- **Fan-out size**: Impact of using 1 vs. 3 vs. 5 generation models.
- **Safety filtering ablation**: Impact of the safety filter on the final candidate pool.
- **Ranking strategy comparison**: LLM judge ranking vs. random selection of safe candidates.
