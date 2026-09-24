# CSHSM Architecture Document

## 1. System Objective
The Counter-Speech Generation for Hate Content on Social Media (CSHSM) system aims to detect hate speech in social media comments and automatically retrieve or generate safe, relevant, and constructive counter-speech responses.

## 2. End-to-End Flow
The architecture is designed as a hybrid retrieval-augmented generation (RAG) pipeline with early-exit mechanisms to optimize latency, cost, and safety. 

```
HATE COMMENT
      ↓
PREPROCESSING
      ↓
XLM-R
      ↓
HATE / NON-HATE
      ↓
NON-HATE → SHORT-CIRCUIT

HATE
  ↓
CATEGORY
  ↓
BGE-M3
  ↓
FAISS
  ↓
TOP-K RESULTS
  ↓
SIMILARITY SCORE
  ↓
CALIBRATED τ
  ↙          ↘
HIGH          LOW
 ↓             ↓
RETRIEVAL    GENERATION
CURATED       FAN-OUT
CANDIDATE       ↓
   |       GENERATION MODELS
   |            ↓
   |        CANDIDATES
   |            ↓
   |    BASIC VALIDATION
   |            ↓
   |      SAFETY FILTER
   |            ↓
   |     LLM EVALUATION
   |            ↓
   |          RANKING
    ↘          ↙
    FINAL VALIDATION
       ↙       ↘
    PASS        FAIL
     ↓            ↓
 RESPONSE    SAFE FALLBACK
```

## 3. Component Responsibilities
- **Preprocessing**: Cleans input text, normalizes emojis, URLs, and mentions.
- **XLM-R Classifier**: Identifies whether the input is hate speech and classifies its category (e.g., racism, sexism).
- **BGE-M3 Embedder**: Converts the categorized hate comment into a dense vector representation.
- **FAISS Vector Store**: Performs fast similarity search against a pre-indexed database of trusted counter-speech responses.
- **Similarity Threshold (τ)**: Routes the request dynamically based on how closely the input matches known counter-speech scenarios.
- **Generation Fan-Out**: Asynchronously queries multiple LLMs to generate diverse counter-speech candidates when retrieval fails.
- **Safety Filter**: Hard-filters explicit toxicity and unaligned outputs using deterministic checks and safety ML models.
- **LLM Evaluator/Ranker**: Scores the safe candidates based on relevance, persuasiveness, and de-escalation, choosing the best one.
- **Safe Fallback**: A hardcoded safe response ("I cannot respond to this respectfully") returned if the entire pipeline fails safety checks.

## 4. Data Flow
- Input strings are parsed into structured requests.
- Flow control explicitly stops if the input is benign.
- Data passed to the generation fan-out includes the original comment, the classified category, and prompt instructions.
- Candidates flow through filtering stages where they are dropped if they fail validation.

## 5. Retrieval Branch (High Similarity)
When the FAISS query returns a Top-K result with a similarity score $\ge$ calibrated $\tau$:
- The system trusts the curated retrieval candidate as a strong initial match.
- The generation fan-out is bypassed, saving compute/API costs.
- The candidate proceeds to Final Validation to ensure it meets safety and context requirements.

## 6. Generation Branch (Low Similarity)
When the FAISS query returns results with similarity scores $<$ calibrated $\tau$:
- The system queries multiple (e.g., 5) distinct generation models concurrently.
- This mitigates the risk of a single model producing a poor or hallucinated counter-speech response for novel hate variants.

## 7. Safety/Validation Flow
Both retrieved and generated responses must ultimately pass Final Validation. No response bypasses the final safety gate.
- Generated candidates undergo *Basic Validation* (length/format) and *Safety Filtering* (toxicity, hate reinforcement, escalation, etc.) *before* ranking.
- The single chosen response from generation, or the curated retrieval candidate, undergoes *Final Validation* as a last line of defense before delivery to the user.

## 8. Routing Logic
The decision to retrieve vs. generate is strictly governed by the empirically calibrated threshold ($\tau$). This is a core research component of the system.

## 9. Failure/Fallback Behavior
If all 5 generated candidates fail safety filtering, or if the chosen candidate fails Final Validation, the system aborts and returns the Safe Fallback. It will *never* return an unverified response.

## 10. External Dependencies
- **Local**: PyTorch, Hugging Face `transformers`, `sentence-transformers`, `faiss-cpu`.
- **Cloud/API**: Multiple LLM providers (e.g., OpenAI, Anthropic, Google) for the generation fan-out and LLM-as-a-judge ranking.

## 11. Interfaces Between Major Components
- **Classifier -> Router**: `(is_hate: bool, category: str)`
- **Embedder -> FAISS**: `dense_vector: np.ndarray`
- **FAISS -> Router**: `[(response_text: str, score: float)]`
- **Fan-out -> Safety**: `List[str]` (Candidates)
- **Safety -> Ranker**: `List[str]` (Safe Candidates)
- **Ranker -> Final**: `str` (Best Response)
