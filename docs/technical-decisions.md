# Technical Decision Register

This document records the currently approved architectural decisions for the CSHSM project. 

### 1. Classification Model
- **Decision**: Use XLM-RoBERTa (XLM-R) for hate and category classification.
- **Reason**: Excellent multilingual performance, lightweight enough for local CPU inference, well-supported in Hugging Face.
- **Alternatives considered**: mBERT (weaker on hate speech), local LLMs (too large for storage limit).
- **Status**: Approved.
- **What evidence may cause reconsideration**: If inference latency is too high on the target hardware, we may downgrade to a distilled model.

### 2. Semantic Embeddings
- **Decision**: Use BGE-M3 for semantic embeddings.
- **Reason**: State-of-the-art multilingual embedding model, capable of capturing semantic nuance.
- **Alternatives considered**: OpenAI `text-embedding-3` (adds unnecessary API dependency and cost for a component that fits locally).
- **Status**: Approved.
- **What evidence may cause reconsideration**: Poor Recall@K during retrieval experiments.

### 3. Counter-Speech Knowledge Base
- **Decision**: Explicitly construct a Counter-Speech Knowledge Base as a distinct phase prior to embedding.
- **Reason**: The quality of retrieval depends entirely on the curation, cleaning, and safety filtering of the underlying trusted responses. It is not merely "a vector database."
- **Alternatives considered**: Directly piping raw datasets into FAISS.
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A.

### 4. Vector Database
- **Decision**: Use FAISS for vector retrieval.
- **Reason**: Standard, highly optimized, CPU-friendly vector similarity search.
- **Alternatives considered**: ChromaDB, Pinecone (unnecessary overhead/external dependency for our dataset size).
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A; FAISS is standard and sufficient.

### 5. Similarity Routing
- **Decision**: Use a calibrated similarity threshold ($\tau$) to route requests.
- **Reason**: Ensures we only rely on the vector database when we are highly confident the match is appropriate for counter-speech.
- **Alternatives considered**: Always retrieve, always generate.
- **Status**: Approved.
- **What evidence may cause reconsideration**: If the similarity score distribution of positive vs. negative pairs overlaps entirely, calibration will be impossible, requiring an ML-based router.

### 6. Routing Strategy
- **Decision**: Retrieval-first routing.
- **Reason**: Trusted human-authored/curated responses are safer, cheaper, and faster than LLM generation.
- **Alternatives considered**: Generation-first, parallel retrieval + generation.
- **Status**: Approved.
- **What evidence may cause reconsideration**: If retrieved responses are consistently scored lower by human evaluators than generated ones.

### 7. Fallback Strategy
- **Decision**: Generation fallback for low-confidence retrieval.
- **Reason**: The static database cannot cover the infinite long-tail of novel hate speech.
- **Alternatives considered**: Silent failure (bad UX).
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A.

### 8. Generation Strategy
- **Decision**: Concurrent generation using multiple models (fan-out).
- **Reason**: Diversifies the candidate pool, mitigating the risk of a single model refusing to answer, hallucinating, or producing poor counter-speech.
- **Alternatives considered**: Single model generation.
- **Status**: Approved.
- **What evidence may cause reconsideration**: API costs or latency becoming prohibitive.

### 9. Safety Validation
- **Decision**: Strict safety filtering of generated candidates.
- **Reason**: Counter-speech systems must not escalate toxicity or generate harmful content under any circumstances.
- **Alternatives considered**: Relying purely on generation model alignments.
- **Status**: Approved.
- **What evidence may cause reconsideration**: High false-positive rate (over-censorship) blocking valid counter-speech.

### 10. Candidate Evaluation
- **Decision**: LLM-based evaluation of candidates.
- **Reason**: Traditional NLP metrics (BLEU, ROUGE) are ineffective for evaluating the quality and nuance of counter-speech.
- **Alternatives considered**: Human-in-the-loop (too slow for real-time).
- **Status**: Approved.
- **What evidence may cause reconsideration**: High cost or latency, or low correlation with human judgments in validation tests.

### 11. Candidate Selection
- **Decision**: Implement a ranking system based on the LLM evaluation.
- **Reason**: We need a deterministic way to pick the best response from the surviving candidate pool.
- **Alternatives considered**: Random selection of safe candidates.
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A.

### 12. Final Validation
- **Decision**: A final validation step for the selected response (whether retrieved or generated).
- **Reason**: An absolute last line of defense against harmful output.
- **Alternatives considered**: Skipping this step to save latency.
- **Status**: Approved.
- **What evidence may cause reconsideration**: If it proves entirely redundant to the prior safety filter.

### 13. Safe Fallback
- **Decision**: System returns a hardcoded safe fallback if validation fails.
- **Reason**: Prevents system crashes or harmful outputs when the pipeline cannot produce a safe response.
- **Alternatives considered**: Returning the original hate comment (dangerous).
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A.

### 14. System Architecture
- **Decision**: Modular monolithic backend rather than unnecessary microservices.
- **Reason**: Simplifies development, testing, and deployment for a research prototype.
- **Alternatives considered**: Microservices for classification, retrieval, and generation.
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A.

### 15. Project Execution
- **Decision**: Milestone-by-milestone implementation.
- **Reason**: Ensures rigorous testing and prevents compounding errors in complex ML pipelines.
- **Alternatives considered**: Big-bang integration.
- **Status**: Approved.
- **What evidence may cause reconsideration**: N/A.
