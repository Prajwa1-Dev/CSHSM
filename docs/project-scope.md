# Project Scope

## In Scope
- **Hate/Non-Hate Detection**: Filtering inputs to ensure counter-speech is only generated for true hate speech.
- **Hate-Category Classification**: Identifying the type of hate to inform embedding and retrieval.
- **Semantic Retrieval**: Using vector similarity to find trusted responses.
- **BGE-M3 Embeddings**: Generating high-quality multilingual dense vectors.
- **FAISS**: Local CPU-based vector indexing and retrieval.
- **Threshold Calibration**: Empirically determining $\tau$ to balance retrieval vs. generation.
- **Counter-Speech Generation**: Prompting LLMs for responses when retrieval fails.
- **Concurrent Generation**: Fanning out to 5 separate generation models simultaneously.
- **Candidate Evaluation**: LLM-as-a-judge scoring of generated responses.
- **Safety Filtering**: Broad evaluation of candidates for toxicity, hate reinforcement, counter-hate, escalation, threats, discriminatory content, and irrelevance. No response bypasses the final safety gate.
- **Ranking**: Selecting the single best counter-speech response based on a specific rubric.
- **Fallback**: Returning a pre-defined safe string if the pipeline fails or all candidates are unsafe.
- **API**: A REST backend exposing the pipeline.
- **Frontend/Demo**: A lightweight interface for human interaction.
- **Evaluation**: Quantitative analysis of the system and its components.
- **Experiments and Ablations**: Statistical comparison of baselines (retrieval-only, generation-only) vs. the hybrid architecture.

## Out of Scope (Unless Later Justified)
- **Training Massive Foundation Models from Scratch**: We will use off-the-shelf, API-hosted, or fine-tuned base models.
- **Production-Scale Social-Media Moderation**: The system is a research prototype, not engineered for Twitter/Facebook production scale throughput (e.g., millions of requests per second).
- **Large-Scale Distributed Infrastructure**: No Kubernetes, Redis clusters, or complex message queues.
- **Unnecessary Microservices**: The system will be a modular monolith to simplify testing and deployment.
- **Unnecessary Cloud Infrastructure**: We will not deploy to AWS/GCP as part of this scope; local hosting + Cloud APIs is sufficient.
- **Features Unrelated to the Research Question**: User accounts, database persistence for user data, or complex UI analytics.

## Final-Year-Project Scope
The goal is to deliver a scientifically valid, rigorously evaluated research prototype. The priority is on demonstrating the effectiveness of the hybrid RAG architecture (calibrated $\tau$, fan-out, LLM evaluation) over basic baselines, rather than building a scalable consumer product. The final deliverable must include the working software, reproducible experiment scripts, and comprehensive evaluation metrics suitable for a research paper.
