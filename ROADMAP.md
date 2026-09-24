# CSHSM — Counter-Speech Generation for Hate Content on Social Media
## Research & Implementation Roadmap

> [!IMPORTANT]
> **Hardware Constraint Acknowledged**: Development is restricted to a MacBook with ~7.75 GB of free disk space. The system architecture explicitly avoids local hosting of large language models (LLMs). Local inference is restricted to lightweight classification and embedding models (XLM-R Base, BGE-M3).

---

## A. Complete Milestone Roadmap

We will build CSHSM in 22 sequential milestones (M0-M21). **No milestone begins until the previous one is fully tested and verified.**

### M0 — Planning & Architecture Freeze
- **Objective**: Establish the project scope, technical decisions, and research plan. (Completed)

### M1 — Environment & Repository Setup
- **Objective**: Initialize Python environment, dependency management, and configuration structure.

### M2 — Dataset Acquisition & Analysis
- **Objective**: Evaluate and select appropriate hate-speech and counter-speech datasets.
- **Tasks**: Analyze relevance, licensing, language coverage, label quality, counter-speech quality, duplication, and leakage. Final selection must be evidence-based.

### M3 — Dataset Preprocessing
- **Objective**: Clean and normalize text, standardize schemas.

### M4 — XLM-R Hate Detection & Classification
- **Objective**: Implement the XLM-RoBERTa pipeline to classify hate vs. non-hate and categorize hate types.
- **Tasks**: Determine whether to use a single classification head, multiple heads, or another design experimentally.

### M5 — Counter-Speech Knowledge Base Construction
- **Objective**: Explicitly construct the Counter-Speech Knowledge Base (prior to vectorization).
- **Tasks**: Source selection, counter-speech extraction, cleaning, categorization, metadata generation, duplicate/near-duplicate handling, safety filtering, quality criteria enforcement, and dataset leakage prevention.

### M6 — BGE-M3 Embedding Pipeline
- **Objective**: Generate multilingual dense vectors for the Knowledge Base.
- **Tasks**: Evaluate whether the embedding query should include the hate comment alone, or the hate comment + predicted category + structured context.

### M7 — FAISS Retrieval System
- **Objective**: Build the vector retrieval index.
- **Tasks**: Benchmark and select the appropriate FAISS index type and similarity metric experimentally.

### M8 — Retrieval Evaluation & τ Calibration
- **Objective**: Empirically calibrate the routing threshold $\tau$.
- **Tasks**: Evaluate retrieval/generation trade-offs on validation data. **Do not use arbitrary threshold values.**

### M9 — Generation Model Integration
- **Objective**: Select and integrate LLM APIs for counter-speech generation.
- **Tasks**: Evaluate and select models based on availability, quality, multilingual capability, latency, cost, API limits, and safety.

### M10 — Concurrent Generation / 5-Model Fan-Out
- **Objective**: Implement asynchronous concurrent calls to the 5 selected models.

### M11 — LLM Evaluation & Ranking
- **Objective**: Implement an LLM-as-a-judge system to score and select the best candidate.

### M12 — Safety & Final Validation
- **Objective**: Implement rigorous safety checks.
- **Tasks**: Filter for toxicity, hate reinforcement, counter-hate, escalation, threats, discriminatory content, and irrelevance. Apply to both generated and retrieved candidates.

### M13 — Hybrid Adaptive Routing
- **Objective**: Wire the XLM-R, FAISS, Calibrated $\tau$, and Generation Fan-Out together into the main pipeline.

### M14 — FastAPI Backend
- **Objective**: Wrap the pipeline in a REST API.

### M15 — Frontend / Demo Interface
- **Objective**: Build a lightweight UI (e.g., Streamlit).

### M16 — End-to-End Integration & Testing
- **Objective**: Verify system functionality from frontend to backend.

### M17 — Baseline & Architecture Comparison
- **Objective**: Execute core experiments.
- **Comparisons**: 
  - Retrieval-only vs Generation-only vs Hybrid
  - 1-model vs 3-model vs 5-model fan-out
  - Calibrated $\tau$ vs uncalibrated/baseline routing

### M18 — Human Evaluation
- **Objective**: Validate LLM-as-a-judge scores against human annotators.

### M19 — Optimization & Ablation Studies
- **Objective**: Measure baseline latency, identify bottlenecks, and perform necessary optimization based on actual benchmarks.

### M20 — Deployment
- **Objective**: Prepare the research prototype for final demonstration (local/API-based).

### M21 — Final Experiments, Documentation & Research Deliverables
- **Objective**: Finalize the research paper/report with accuracy, safety, latency, cost, and failure rate metrics.

---

## B. Core Architecture

```text
Hate Comment
    ↓
Preprocessing
    ↓
XLM-R
    ↓
Hate / Non-Hate
    ↓
Non-Hate → Short Circuit

Hate
    ↓
Category
    ↓
BGE-M3
    ↓
FAISS
    ↓
Top-K
    ↓
Retrieval Score
    ↓
Calibrated τ
    ↙          ↘
  HIGH         LOW
   ↓            ↓
RETRIEVAL    GENERATION
```

---

## C. Technology Stack & Definitions
- **Project Name**: Counter-Speech Generation for Hate Content on Social Media
- **Classification**: XLM-R
- **Embedding**: BGE-M3
- **Vector Store**: FAISS
- **Scope**: Final-year research project prototype (not a production-scale distributed system).
