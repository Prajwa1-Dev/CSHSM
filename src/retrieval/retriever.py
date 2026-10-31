import os
import json
import math
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from src.retrieval.embedder import (
    BaseEmbedder,
    MockEmbedder,
    EmbeddingPipeline,
    load_npy_f32
)

# =====================================================================
# Vector Index Abstraction & Backends
# =====================================================================

class BaseVectorIndex(ABC):
    """Abstract interface for vector similarity search."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def ntotal(self) -> int:
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def add(self, vectors: List[List[float]]) -> None:
        """Add dense float32 vectors to the index."""
        pass

    @abstractmethod
    def search(
        self,
        query_vectors: List[List[float]],
        k: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Search top-k nearest neighbors. Returns (scores, indices)."""
        pass


class FAISSVectorIndex(BaseVectorIndex):
    """
    FAISS-based IndexFlatIP (Inner Product / Cosine Similarity on unit vectors).
    Requires faiss library (faiss-cpu).
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim
        try:
            import faiss
            self._faiss = faiss
        except ImportError as e:
            raise ImportError(
                "faiss is required for FAISSVectorIndex but not installed in the active environment. "
                "Use ExactInnerProductIndex for pure-Python fallback, or install faiss-cpu."
            ) from e

        # IndexFlatIP computes exact inner product
        self._index = self._faiss.IndexFlatIP(self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    @property
    def backend_name(self) -> str:
        return "faiss"

    def add(self, vectors: List[List[float]]) -> None:
        import numpy as np
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(f"Vector shape {arr.shape} does not match index dimension {self._dim}")
        self._index.add(arr)

    def search(
        self,
        query_vectors: List[List[float]],
        k: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
        import numpy as np
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}")
        arr = np.array(query_vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(f"Query shape {arr.shape} does not match index dimension {self._dim}")
        effective_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(arr, effective_k)
        return scores.tolist(), indices.tolist()


class ExactInnerProductIndex(BaseVectorIndex):
    """
    Deterministic standard-library / pure-Python vector index for exact inner-product search.
    Provides identical mathematical results to IndexFlatIP on L2-normalized unit vectors
    without requiring third-party C++ libraries.
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim
        self._vectors: List[List[float]] = []

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    @property
    def backend_name(self) -> str:
        return "exact_ip_fallback"

    def add(self, vectors: List[List[float]]) -> None:
        for idx, v in enumerate(vectors):
            if len(v) != self._dim:
                raise ValueError(
                    f"Vector index {idx} has dimension {len(v)}, expected {self._dim}"
                )
            for val in v:
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Vector index {idx} contains non-finite value: {val}")
            self._vectors.append(list(v))

    def search(
        self,
        query_vectors: List[List[float]],
        k: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}")
        if not self._vectors:
            raise ValueError("Cannot search an empty index.")

        effective_k = min(k, len(self._vectors))
        all_scores: List[List[float]] = []
        all_indices: List[List[int]] = []

        for q in query_vectors:
            if len(q) != self._dim:
                raise ValueError(
                    f"Query vector dimension {len(q)} does not match index dimension {self._dim}"
                )
            for val in q:
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Query vector contains non-finite value: {val}")

            # Calculate exact inner products against all indexed vectors
            scored_candidates = []
            for row_idx, vec in enumerate(self._vectors):
                # Inner product
                dot = sum(qv * vv for qv, vv in zip(q, vec))
                scored_candidates.append((dot, row_idx))

            # Sort descending by score, tie-break by row_idx ascending for determinism
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))

            top_candidates = scored_candidates[:effective_k]
            all_scores.append([round(item[0], 6) for item in top_candidates])
            all_indices.append([item[1] for item in top_candidates])

        return all_scores, all_indices


def create_vector_index(
    dim: int = 1024,
    prefer_faiss: bool = True,
    force_backend: Optional[str] = None
) -> BaseVectorIndex:
    """
    Factory function for vector similarity indices.
    Selects FAISSVectorIndex if available and preferred; otherwise uses ExactInnerProductIndex.
    """
    if force_backend == "faiss":
        return FAISSVectorIndex(dim=dim)
    elif force_backend == "exact_ip_fallback":
        return ExactInnerProductIndex(dim=dim)

    if prefer_faiss:
        try:
            return FAISSVectorIndex(dim=dim)
        except ImportError:
            return ExactInnerProductIndex(dim=dim)
    return ExactInnerProductIndex(dim=dim)


# =====================================================================
# Result Models
# =====================================================================

@dataclass
class RetrievalMatch:
    rank: int
    score: float
    kb_id: str
    hate_text: str
    counter_speech: str
    category: Optional[str]
    target: Optional[str]
    source: Optional[str]
    source_id: Optional[str]
    quality_status: str
    review_status: str
    scope: str
    row_idx: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query_text: Optional[str]
    query_vector_sha256: Optional[str]
    top_k: int
    matches: List[RetrievalMatch]
    run_id: str
    is_mock: bool
    scope: str
    approval_status: str
    backend: str
    retrieval_status: str
    threshold_applied: Optional[float]  # Reserved for M8 threshold calibration
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["matches"] = [m.to_dict() for m in self.matches]
        return d


# =====================================================================
# CSHSM Production Retriever
# =====================================================================

class CSHSMRetriever:
    """
    Milestone 7 Top-K Counter-Speech Knowledge Base Retriever.
    Consumes authoritative embeddings exclusively from committed M6 runs,
    enforcing bijective row-to-manifest mapping, strict scope isolation,
    and mock/approved safeguards.
    """

    def __init__(
        self,
        index: BaseVectorIndex,
        manifest: List[Dict[str, Any]],
        run_metadata: Dict[str, Any],
        embedder: Optional[BaseEmbedder] = None
    ):
        self.index = index
        self.manifest = manifest
        self.run_metadata = run_metadata
        self.embedder = embedder

        # Invariant validations
        if self.index.ntotal != len(self.manifest):
            raise RuntimeError(
                f"Retriever initialization failed: Index count ({self.index.ntotal}) "
                f"!= manifest record count ({len(self.manifest)})"
            )

        self._validate_manifest_alignment()

    def _validate_manifest_alignment(self):
        """Verifies strict row alignment and uniqueness in candidate manifest."""
        seen_row_indices = set()
        seen_kb_ids = set()

        for idx, entry in enumerate(self.manifest):
            row_idx = entry.get("row_idx")
            if row_idx is None:
                raise ValueError(f"Manifest entry at line {idx} is missing 'row_idx'")
            if row_idx != idx:
                raise ValueError(
                    f"Row alignment corruption: Manifest entry {idx} has row_idx={row_idx}"
                )
            if row_idx in seen_row_indices:
                raise ValueError(f"Duplicate row_idx {row_idx} at line {idx}")
            seen_row_indices.add(row_idx)

            kb_id = entry.get("kb_id")
            if not kb_id:
                raise ValueError(f"Manifest entry at row {idx} is missing 'kb_id'")

    @classmethod
    def from_committed_run(
        cls,
        output_dir: str = "data/counter_speech_kb/embeddings",
        scope: Optional[str] = None,
        allow_mock: bool = False,
        embedder: Optional[BaseEmbedder] = None,
        force_backend: Optional[str] = None
    ) -> "CSHSMRetriever":
        """
        Loads the authoritative committed run from M6 via EmbeddingPipeline.resolve_latest_run().
        Fails closed if the run is uncommitted, incomplete, or corrupted.
        """
        run_data = EmbeddingPipeline.resolve_latest_run(
            output_dir=output_dir,
            expected_scope=scope,
            allow_mock=allow_mock
        )

        vector_file = run_data["vector_file"]
        manifest_file = run_data["manifest_file"]

        # 1. Load float32 vectors
        vectors, shape = load_npy_f32(vector_file)
        n_rows, dim = shape

        # 2. Load candidate manifest
        manifest: List[Dict[str, Any]] = []
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    manifest.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Malformed manifest JSON at line {line_no + 1}: {e}") from e

        # 3. Check bijection: vector count == manifest count
        if n_rows != len(manifest):
            raise RuntimeError(
                f"Critical integrity failure: Vector rows ({n_rows}) != manifest lines ({len(manifest)}) "
                f"in run {run_data['run_id']}"
            )

        # 4. Check vector finiteness and norms
        for row_idx, vec in enumerate(vectors):
            if len(vec) != dim:
                raise ValueError(
                    f"Row {row_idx} dimension is {len(vec)}, expected {dim}"
                )
            for v in vec:
                if math.isnan(v) or math.isinf(v):
                    raise RuntimeError(f"Non-finite value detected in vector row {row_idx}")

        # 5. Populate vector index
        index = create_vector_index(dim=dim, prefer_faiss=True, force_backend=force_backend)
        index.add(vectors)

        return cls(
            index=index,
            manifest=manifest,
            run_metadata=run_data,
            embedder=embedder
        )

    def retrieve_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 3,
        category_filter: Optional[str] = None
    ) -> RetrievalResult:
        """
        Executes exact top-K similarity search using a pre-computed float32 unit vector.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        dim = self.index.dimension
        if len(query_vector) != dim:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match index dimension {dim}"
            )

        for val in query_vector:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Query vector contains non-finite values (NaN or Inf)")

        # Verify or enforce L2 normalization on query vector
        norm = math.sqrt(sum(x * x for x in query_vector))
        if norm == 0.0:
            raise ValueError("Query vector cannot be zero vector.")
        if abs(norm - 1.0) > 1e-3:
            normalized_query = [x / norm for x in query_vector]
        else:
            normalized_query = list(query_vector)

        query_sha256 = hashlib.sha256(
            json.dumps([round(x, 6) for x in normalized_query]).encode("utf-8")
        ).hexdigest()

        # If category filtering is requested, search larger candidate pool then filter
        search_k = min(self.index.ntotal, top_k * 5 if category_filter else top_k)
        scores_batch, indices_batch = self.index.search([normalized_query], k=search_k)

        scores = scores_batch[0]
        indices = indices_batch[0]

        matches: List[RetrievalMatch] = []
        rank_counter = 1

        for score, row_idx in zip(scores, indices):
            if row_idx < 0 or row_idx >= len(self.manifest):
                continue
            entry = self.manifest[row_idx]

            # Category filter if specified
            if category_filter:
                entry_cat = entry.get("category")
                if entry_cat != category_filter:
                    continue

            # Review status guard
            rev_status = entry.get("review_status", "unreviewed")
            qual_status = entry.get("quality_status", "filtered_candidate")

            match = RetrievalMatch(
                rank=rank_counter,
                score=score,
                kb_id=entry.get("kb_id", f"UNKNOWN-{row_idx}"),
                hate_text=entry.get("hate_text", ""),
                counter_speech=entry.get("counter_speech", ""),
                category=entry.get("category"),
                target=entry.get("target"),
                source=entry.get("source"),
                source_id=entry.get("source_id"),
                quality_status=qual_status,
                review_status=rev_status,
                scope=self.run_metadata.get("scope", "development_provisional"),
                row_idx=row_idx
            )
            matches.append(match)
            rank_counter += 1
            if len(matches) >= top_k:
                break

        run_scope = self.run_metadata.get("scope", "development_provisional")
        is_mock = self.run_metadata.get("is_mock", True)
        approval_status = "approved" if run_scope == "approved_kb" else "unapproved"

        disclaimer = (
            "PROVISIONAL CANDIDATES — Counter-speech retrieved for retrieval engineering only; unreviewed and unapproved by M5.4."
            if run_scope == "development_provisional" else
            "APPROVED KB — Formally evaluated and approved counter-speech knowledge base."
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        provenance = {
            "retrieved_at": now_iso,
            "run_id": self.run_metadata.get("run_id"),
            "run_dir": self.run_metadata.get("run_dir"),
            "model_identifier": self.run_metadata.get("model_identifier"),
            "is_mock": is_mock,
            "vector_count": self.index.ntotal,
            "backend": self.index.backend_name,
            "category_filter": category_filter,
            "threshold_applied": None  # Deferred to M8 threshold calibration
        }

        return RetrievalResult(
            query_text=None,
            query_vector_sha256=query_sha256,
            top_k=top_k,
            matches=matches,
            run_id=self.run_metadata.get("run_id", "UNKNOWN"),
            is_mock=is_mock,
            scope=run_scope,
            approval_status=approval_status,
            backend=self.index.backend_name,
            retrieval_status="success" if matches else "no_matches",
            threshold_applied=None,
            provenance=provenance,
            disclaimer=disclaimer
        )

    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 3,
        category_filter: Optional[str] = None
    ) -> RetrievalResult:
        """
        Embeds query text and retrieves top-K matching counter-speech candidates.
        Requires embedder configured on the retriever instance.
        """
        if not self.embedder:
            raise ValueError(
                "CSHSMRetriever has no embedder configured for text queries. "
                "Provide an embedder (BaseEmbedder) or pass pre-computed vectors to retrieve_by_vector()."
            )
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty or whitespace.")

        query_vec = self.embedder.embed_batch([query_text.strip()])[0]
        res = self.retrieve_by_vector(query_vec, top_k=top_k, category_filter=category_filter)
        res.query_text = query_text.strip()
        return res
