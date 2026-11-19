import os
# Default to offline Hugging Face mode to ensure fast, socket-free local model loading
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
import json
import math
import array
import struct
import shutil
import hashlib
import gc
import uuid
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any

from src.kb.schema import KBEntry, validate_entry

# =====================================================================
# Low-level .npy binary format serializer / deserializer
# Works with NumPy if installed, or standard library array/struct.
# =====================================================================

def save_npy_f32(filepath: str, data: List[List[float]], shape: Tuple[int, int]):
    """
    Saves a 2D float32 array in standard NumPy .npy format (v1.0).
    Uses numpy.save if available, otherwise pure standard library.
    """
    try:
        import numpy as np
        arr = np.array(data, dtype=np.float32)
        with open(filepath, "wb") as f:
           np.save(f, arr)
        return
    except ImportError:
        pass

    magic = b"\x93NUMPY\x01\x00"
    header_str = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}"
    header_bytes = header_str.encode("latin1")
    total_len = len(magic) + 2 + len(header_bytes) + 1
    padding = (64 - (total_len % 64)) % 64
    header_bytes += b" " * padding + b"\n"
    header_len = struct.pack("<H", len(header_bytes))

    with open(filepath, "wb") as f:
        f.write(magic)
        f.write(header_len)
        f.write(header_bytes)
        flat_data = [float(val) for row in data for val in row]
        arr = array.array("f", flat_data)
        f.write(arr.tobytes())

def load_npy_f32(filepath: str) -> Tuple[List[List[float]], Tuple[int, int]]:
    """
    Loads a 2D float32 array from a NumPy .npy format file.
    Returns (data_list_of_lists, shape).
    """
    try:
        import numpy as np
        arr = np.load(filepath)
        return arr.tolist(), arr.shape
    except ImportError:
        pass

    with open(filepath, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Invalid .npy file format: {filepath}")
        version = f.read(2)
        h_len = struct.unpack("<H", f.read(2))[0]
        header_bytes = f.read(h_len)
        header_str = header_bytes.decode("latin1")

        # Parse shape from header string
        shape_match = re.search(r"'shape':\s*\(([^)]*)\)", header_str)
        if not shape_match:
            raise ValueError(f"Cannot parse shape from .npy header: {header_str}")
        shape_parts = [p.strip() for p in shape_match.group(1).split(",") if p.strip()]
        shape = tuple(int(p) for p in shape_parts)

        # Read float32 payload
        payload_bytes = f.read()
        num_floats = len(payload_bytes) // 4
        arr = array.array("f")
        arr.frombytes(payload_bytes)

        if len(shape) == 2:
            n_rows, n_cols = shape
            data = [list(arr[i * n_cols : (i + 1) * n_cols]) for i in range(n_rows)]
            return data, shape
        return [list(arr)], shape

# =====================================================================
# Embedder Base and Concrete Classes
# =====================================================================

class BaseEmbedder(ABC):
    """Abstract base class for CSHSM embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def model_identifier(self) -> str:
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute dense vectors for a batch of strings, returned L2-normalized."""
        pass

    def embed_text(self, text: str) -> List[float]:
        """Compute dense vector for a single string; returned L2-normalized."""
        results = self.embed_batch([text])
        return results[0] if results else []


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for tests and CI.
    Generates 1024-dimensional float32 unit vectors deterministically derived
    from the SHA-256 hash of the input text.

    CRITICAL: Output is explicitly labeled as mock and must never be accepted
    as real BGE-M3 outputs.
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_identifier(self) -> str:
        return "MOCK-EMBEDDER-TEST-ONLY"

    @property
    def is_mock(self) -> bool:
        return True

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Deterministic pseudo-random generation using hash digest
            h = hashlib.sha256(text.encode("utf-8")).digest()
            raw_vec = []
            for i in range(self._dim):
                byte_val = h[(i * 3) % len(h)]
                alt_val = h[(i * 7 + 1) % len(h)]
                val = ((byte_val ^ alt_val) / 255.0) - 0.5
                raw_vec.append(val)

            # Compute exact L2 norm
            norm = math.sqrt(sum(v * v for v in raw_vec))
            if norm == 0.0:
                norm = 1.0
            norm_vec = [v / norm for v in raw_vec]
            embeddings.append(norm_vec)
        return embeddings


class BGE_M3_Embedder(BaseEmbedder):
    """
    Production embedding adapter for BAAI/bge-m3 dense representation.
    Requires sentence-transformers and PyTorch.
    Fails with an informative ImportError if dependencies are missing.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 512,
        revision: Optional[str] = None,
        local_files_only: bool = True
    ):
        self._model_name = model_name
        self._device_arg = device
        self._device = None
        self._batch_size = batch_size
        self._max_length = max_length
        self._revision = revision
        self._local_files_only = local_files_only
        self._model = None
        self._dim = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_identifier(self) -> str:
        return self._model_name

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        if self._device_arg is not None:
            return self._device_arg
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for real BGE-M3 embedding generation. "
                "Install with: pip install sentence-transformers"
            ) from e

        import torch
        if self._device_arg is None or self._device_arg == "cuda":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_arg

        kwargs = {"device": self._device}
        if self._revision:
            kwargs["revision"] = self._revision

        if self._local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            kwargs["model_kwargs"] = {"local_files_only": True}

        try:
            self._model = SentenceTransformer(self._model_name, **kwargs)
        except Exception:
            if "model_kwargs" in kwargs:
                kwargs.pop("model_kwargs", None)
            self._model = SentenceTransformer(self._model_name, **kwargs)

        self._model.max_seq_length = self._max_length

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        import torch
        cleaned_texts = [t if (t and t.strip()) else " " for t in texts]
        with torch.no_grad():
            outputs = self._model.encode(
                cleaned_texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return outputs.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Compute dense vector for a single string; returned L2-normalized."""
        results = self.embed_batch([text])
        return results[0] if results else []

    def embed_query(self, query: str) -> List[float]:
        """Convenience query embedding method."""
        return self.embed_text(query)

    def embed_document(self, doc: str) -> List[float]:
        """Convenience document embedding method."""
        return self.embed_text(doc)


# =====================================================================
# Embedding Pipeline & Artifact Management
# =====================================================================

class EmbeddingPipeline:
    """
    Orchestrates candidate loading, text formatting, batch embedding,
    manifest creation, integrity checking, and atomic persistence.
    """

    VALID_SCOPES = ["development_provisional", "approved_kb"]
    VALID_STRATEGIES = ["hate_text", "structured_category_hate"]
    RUNS_SUBDIR = "runs"
    LATEST_POINTER_FILE = "LATEST.json"
    COMMIT_MARKER_FILE = "COMMITTED"

    def __init__(
        self,
        embedder: BaseEmbedder,
        scope: str = "development_provisional",
        strategy: str = "hate_text",
        batch_size: int = 4,
        dev_limit: Optional[int] = None,
        output_dir: str = "data/counter_speech_kb/embeddings",
        reports_dir: str = "reports"
    ):
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"Invalid scope '{scope}'. Allowed: {self.VALID_SCOPES}")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{strategy}'. Allowed: {self.VALID_STRATEGIES}")

        self.embedder = embedder
        self.scope = scope
        self.strategy = strategy
        self.batch_size = max(1, batch_size)
        self.dev_limit = dev_limit
        self.output_dir = output_dir
        self.reports_dir = reports_dir

    @staticmethod
    def normalize_text(text: str) -> str:
        """Conservative whitespace collapse and trim."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def format_text(self, entry: KBEntry) -> str:
        """Format text based on chosen embedding strategy."""
        ht = self.normalize_text(entry.hate_text)
        if not ht:
            return ""
        if self.strategy == "structured_category_hate":
            cat = entry.category if entry.category else "Unspecified"
            return f"Category: {cat} | Hate: {ht}"
        return ht

    def _validate_scope_invariants(self, entries: List[KBEntry], input_path: str):
        """
        Enforce fail-closed rules depending on execution scope.
        """
        if self.scope == "approved_kb":
            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"Approved KB mode fail-closed: Input file not found at {input_path}"
                )
            if not entries:
                raise ValueError(
                    f"Approved KB mode fail-closed: Input file {input_path} contains zero valid entries."
                )
            for idx, e in enumerate(entries):
                if e.quality_status != "approved" or e.review_status not in ("approved", "human-approved"):
                    raise ValueError(
                        f"Approved KB mode fail-closed: Entry index {idx} ({e.kb_id}) has "
                        f"quality_status='{e.quality_status}', review_status='{e.review_status}'. "
                        "All records in approved_kb scope must be genuinely approved by M5.4."
                    )
        elif self.scope == "development_provisional":
            # Allowed to proceed on filtered_candidates, but records unapproved status explicitly
            pass

    def run(self, input_path: str) -> Dict[str, Any]:
        """
        Executes embedding generation over the specified KB dataset.
        Returns the audit report dictionary.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input KB file not found: {input_path}")

        with open(input_path, "rb") as f:
            input_sha256 = hashlib.sha256(f.read()).hexdigest()

        # 1. Load and validate records
        raw_entries = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    entry = KBEntry.from_dict(record) if hasattr(KBEntry, "from_dict") else KBEntry(**record)
                    raw_entries.append(entry)
                except Exception as e:
                    raise ValueError(f"Malformed JSONL at line {line_no + 1} of {input_path}: {e}") from e

        # 2. Scope invariant checks
        self._validate_scope_invariants(raw_entries, input_path)

        # 3. Filter valid and extract texts
        valid_entries = []
        skipped_records = []

        for e in raw_entries:
            if not validate_entry(e):
                skipped_records.append({"kb_id": e.kb_id, "reason": "Schema validation failed"})
                continue
            formatted = self.format_text(e)
            if not formatted:
                skipped_records.append({"kb_id": e.kb_id, "reason": "Empty hate_text after normalization"})
                continue
            valid_entries.append((e, formatted))

        # 4. Apply dev_limit if configured
        if self.dev_limit is not None and self.dev_limit > 0:
            valid_entries = valid_entries[: self.dev_limit]

        total_to_embed = len(valid_entries)
        if total_to_embed == 0:
            raise ValueError(f"No valid records available to embed from {input_path}")

        # 5. Generate embeddings in conservative batches
        all_embeddings: List[List[float]] = []
        manifest_records: List[Dict[str, Any]] = []

        for start_idx in range(0, total_to_embed, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_to_embed)
            batch = valid_entries[start_idx:end_idx]
            batch_texts = [text for _, text in batch]

            batch_vecs = self.embedder.embed_batch(batch_texts)

            for offset, (entry, text) in enumerate(batch):
                row_idx = start_idx + offset
                vec = batch_vecs[offset]
                all_embeddings.append(vec)

                manifest_records.append({
                    "row_idx": row_idx,
                    "kb_id": entry.kb_id,
                    "hate_text": entry.hate_text,
                    "counter_speech": entry.counter_speech,
                    "category": entry.category,
                    "target": entry.target,
                    "source": entry.source,
                    "source_id": entry.source_id,
                    "quality_status": entry.quality_status,
                    "scope": self.scope,
                    "text_embedded": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "vector_dim": self.embedder.dimension
                })

        # 6. Integrity checks
        matrix_shape = (len(all_embeddings), self.embedder.dimension)
        if matrix_shape[0] != len(manifest_records):
            raise RuntimeError(
                f"Integrity check failed: Vector rows ({matrix_shape[0]}) != manifest rows ({len(manifest_records)})"
            )

        for row_idx, vec in enumerate(all_embeddings):
            if len(vec) != self.embedder.dimension:
                raise RuntimeError(
                    f"Integrity check failed: Row {row_idx} dimension is {len(vec)}, expected {self.embedder.dimension}"
                )
            for v in vec:
                if math.isnan(v) or math.isinf(v):
                    raise RuntimeError(f"Integrity check failed: Non-finite value at row {row_idx}")
            norm = math.sqrt(sum(v * v for v in vec))
            if abs(norm - 1.0) > 1e-4:
                raise RuntimeError(
                    f"Integrity check failed: Row {row_idx} L2 norm is {norm:.6f}, expected 1.0"
                )

        # 7. Pre-run backup before modifying any active published artifacts
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        backup_dir = self._backup_existing_artifacts()

        # 8. Setup isolated versioned run directory under self.output_dir/runs/<run_id>
        prefix = "bge_m3_mock" if self.embedder.is_mock else "bge_m3"
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_uuid = uuid.uuid4().hex[:8]
        run_id = f"{prefix}_{run_ts}_{run_uuid}"
        runs_dir = os.path.join(self.output_dir, self.RUNS_SUBDIR)
        os.makedirs(runs_dir, exist_ok=True)
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        run_vec_path = os.path.join(run_dir, f"{prefix}_dense.npy")
        run_manifest_path = os.path.join(run_dir, f"{prefix}_manifest.jsonl")
        run_report_path = os.path.join(run_dir, "m6_embedding_report.json")
        run_skipped_path = os.path.join(run_dir, "m6_skipped_records.jsonl")
        run_commit_path = os.path.join(run_dir, self.COMMIT_MARKER_FILE)
        latest_pointer_path = os.path.join(self.output_dir, self.LATEST_POINTER_FILE)

        tmp_vec = run_vec_path + ".tmp"
        tmp_manifest = run_manifest_path + ".tmp"
        tmp_skipped = run_skipped_path + ".tmp"
        tmp_report = run_report_path + ".tmp"
        tmp_commit = run_commit_path + ".tmp"
        tmp_latest = latest_pointer_path + ".tmp"

        try:
            # 8a. Write vector matrix inside run_dir
            save_npy_f32(tmp_vec, all_embeddings, matrix_shape)
            os.replace(tmp_vec, run_vec_path)

            # 8b. Write candidate manifest inside run_dir
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                for rec in manifest_records:
                    rec["run_id"] = run_id
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_manifest, run_manifest_path)

            # 8c. Write skipped records inside run_dir if any
            if skipped_records:
                with open(tmp_skipped, "w", encoding="utf-8") as f:
                    for s in skipped_records:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                os.replace(tmp_skipped, run_skipped_path)

            # 8d. Compute cryptographic hashes for all generated run files
            with open(run_vec_path, "rb") as f:
                vec_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(run_manifest_path, "rb") as f:
                manifest_sha256 = hashlib.sha256(f.read()).hexdigest()
            skipped_sha256 = None
            if skipped_records and os.path.exists(run_skipped_path):
                with open(run_skipped_path, "rb") as f:
                    skipped_sha256 = hashlib.sha256(f.read()).hexdigest()

            # 8e. Build and write metadata report inside run_dir
            now_iso = datetime.now(timezone.utc).isoformat()
            scope_disclaimer = (
                "PROVISIONAL CANDIDATES — Embedded for retrieval development only; counter-speech remains unapproved by M5.4."
                if self.scope == "development_provisional" else
                "APPROVED KB — Formally evaluated and approved counter-speech knowledge base."
            )
            mock_note = "MOCK EMBEDDINGS — Strictly for tests and engineering validation." if self.embedder.is_mock else None

            artifacts_map = {
                "vector_file": f"{prefix}_dense.npy",
                "vector_sha256": vec_sha256,
                "manifest_file": f"{prefix}_manifest.jsonl",
                "manifest_sha256": manifest_sha256,
                "report_file": "m6_embedding_report.json",
                "commit_marker": self.COMMIT_MARKER_FILE,
                "latest_pointer": self.LATEST_POINTER_FILE,
                "authoritative_run_dir": run_dir,
            }
            if skipped_sha256 is not None:
                artifacts_map["skipped_file"] = "m6_skipped_records.jsonl"
                artifacts_map["skipped_sha256"] = skipped_sha256

            report = {
                "timestamp": now_iso,
                "run_id": run_id,
                "run_dir": os.path.relpath(run_dir, self.output_dir),
                "model_identifier": self.embedder.model_identifier,
                "is_mock": self.embedder.is_mock,
                "embedding_dimension": self.embedder.dimension,
                "normalization": "L2",
                "strategy": self.strategy,
                "kb_scope": self.scope,
                "approval_status": "approved" if self.scope == "approved_kb" else "unapproved",
                "batch_size": self.batch_size,
                "input_file": input_path,
                "input_file_sha256": input_sha256,
                "raw_record_count": len(raw_entries),
                "skipped_record_count": len(skipped_records),
                "embedded_record_count": total_to_embed,
                "matrix_shape": list(matrix_shape),
                "artifacts": artifacts_map,
                "disclaimer": scope_disclaimer,
                "mock_notice": mock_note,
                "publication_status": "committed"
            }

            with open(tmp_report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            os.replace(tmp_report, run_report_path)

            with open(run_report_path, "rb") as f:
                report_sha256 = hashlib.sha256(f.read()).hexdigest()
            artifacts_map["report_sha256"] = report_sha256

            # 9. Validate the complete artifact set on disk before committing
            for p, name in [(run_vec_path, "vector"), (run_manifest_path, "manifest"), (run_report_path, "report")]:
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    raise RuntimeError(f"Artifact verification failed in {run_dir}: {name} is missing or empty.")
            if skipped_records:
                if not os.path.exists(run_skipped_path) or os.path.getsize(run_skipped_path) == 0:
                    raise RuntimeError(f"Artifact verification failed in {run_dir}: skipped-records log is missing or empty.")

            # 10. Seal the run with the COMMITTED marker (written last in run_dir)
            commit_payload = {
                "run_id": run_id,
                "committed_at": now_iso,
                "scope": self.scope,
                "is_mock": self.embedder.is_mock,
                "model_identifier": self.embedder.model_identifier,
                "vector_count": len(all_embeddings),
                "manifest_count": len(manifest_records),
                "matrix_shape": list(matrix_shape),
                "artifacts": artifacts_map
            }
            with open(tmp_commit, "w", encoding="utf-8") as f:
                json.dump(commit_payload, f, indent=2)
            os.replace(tmp_commit, run_commit_path)

            # 11. Atomically update authoritative LATEST.json pointer
            pointer_payload = {
                "run_id": run_id,
                "run_dir": os.path.relpath(run_dir, self.output_dir),
                "scope": self.scope,
                "is_mock": self.embedder.is_mock,
                "model_identifier": self.embedder.model_identifier,
                "vector_count": len(all_embeddings),
                "manifest_count": len(manifest_records),
                "matrix_shape": list(matrix_shape),
                "committed_at": now_iso,
                "commit_marker": self.COMMIT_MARKER_FILE,
                "artifacts": artifacts_map
            }
            with open(tmp_latest, "w", encoding="utf-8") as f:
                json.dump(pointer_payload, f, indent=2)
            os.replace(tmp_latest, latest_pointer_path)

        except Exception as e:
            # Clean up temporary staging files on error
            for p in [tmp_vec, tmp_manifest, tmp_skipped, tmp_report, tmp_commit, tmp_latest]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if os.path.exists(run_commit_path):
                try:
                    os.remove(run_commit_path)
                except OSError:
                    pass
            raise RuntimeError(f"Embedding pipeline failed and aborted safely: {e}") from e

        # 12. Reversible canonical mirror publication (for backward compatibility)
        try:
            self._publish_canonical_mirror(run_dir, prefix, skipped_records, report)
        except Exception as e:
            if backup_dir and os.path.exists(backup_dir):
                self._restore_from_backup(backup_dir)
            raise RuntimeError(f"Canonical mirror publication failed; state rolled back: {e}") from e

        return report

    def _publish_canonical_mirror(
        self,
        run_dir: str,
        prefix: str,
        skipped_records: List[Dict[str, Any]],
        report: Dict[str, Any]
    ):
        """
        Updates canonical compatibility mirror files with rollback guard.
        Authoritative publication remains the committed run_dir referenced by LATEST.json.
        """
        vec_path = os.path.join(self.output_dir, f"{prefix}_dense.npy")
        manifest_path = os.path.join(self.output_dir, f"{prefix}_manifest.jsonl")
        report_path = os.path.join(self.reports_dir, "m6_embedding_report.json")
        skipped_path = os.path.join(self.reports_dir, "m6_skipped_records.jsonl")

        tmp_vec = vec_path + ".tmp"
        tmp_manifest = manifest_path + ".tmp"
        tmp_report = report_path + ".tmp"
        tmp_skipped = skipped_path + ".tmp"

        src_vec = os.path.join(run_dir, f"{prefix}_dense.npy")
        src_manifest = os.path.join(run_dir, f"{prefix}_manifest.jsonl")
        src_report = os.path.join(run_dir, "m6_embedding_report.json")
        src_skipped = os.path.join(run_dir, "m6_skipped_records.jsonl")

        shutil.copy2(src_vec, tmp_vec)
        shutil.copy2(src_manifest, tmp_manifest)
        shutil.copy2(src_report, tmp_report)
        if skipped_records and os.path.exists(src_skipped):
            shutil.copy2(src_skipped, tmp_skipped)

        os.replace(tmp_vec, vec_path)
        os.replace(tmp_manifest, manifest_path)
        if skipped_records and os.path.exists(src_skipped):
            os.replace(tmp_skipped, skipped_path)
        elif os.path.exists(skipped_path):
            os.remove(skipped_path)
        os.replace(tmp_report, report_path)

    def _backup_existing_artifacts(self) -> Optional[str]:
        """
        Safely stages and archives existing published artifacts before any modification.
        Aborts completely if copying fails, leaving active artifacts untouched.
        """
        prefix = "bge_m3_mock" if self.embedder.is_mock else "bge_m3"
        files = [
            os.path.join(self.output_dir, f"{prefix}_dense.npy"),
            os.path.join(self.output_dir, f"{prefix}_manifest.jsonl"),
            os.path.join(self.reports_dir, "m6_embedding_report.json"),
            os.path.join(self.reports_dir, "m6_skipped_records.jsonl"),
            os.path.join(self.output_dir, self.LATEST_POINTER_FILE),
        ]
        existing_files = [f for f in files if os.path.exists(f)]
        if not existing_files:
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.reports_dir, "backups", f"m6_pre_run_{timestamp}")
        staging_dir = backup_dir + ".tmp"

        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)

        try:
            for fpath in existing_files:
                dest = os.path.join(staging_dir, os.path.basename(fpath))
                shutil.copy2(fpath, dest)

            backup_manifest = {
                "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                "backed_up_files": [os.path.basename(f) for f in existing_files],
                "status": "complete"
            }
            with open(os.path.join(staging_dir, "BACKUP_COMPLETE.json"), "w", encoding="utf-8") as f:
                json.dump(backup_manifest, f, indent=2)

            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)
            os.replace(staging_dir, backup_dir)
            return backup_dir

        except Exception as e:
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)
            raise RuntimeError(f"Pre-run backup failed and aborted safely: {e}") from e

    def _restore_from_backup(self, backup_dir: str):
        """
        Restores canonical compatibility mirror files from backup in case of mirror failure.
        """
        if not os.path.exists(backup_dir):
            return
        prefix = "bge_m3_mock" if self.embedder.is_mock else "bge_m3"
        mapping = {
            f"{prefix}_dense.npy": os.path.join(self.output_dir, f"{prefix}_dense.npy"),
            f"{prefix}_manifest.jsonl": os.path.join(self.output_dir, f"{prefix}_manifest.jsonl"),
            "m6_embedding_report.json": os.path.join(self.reports_dir, "m6_embedding_report.json"),
            "m6_skipped_records.jsonl": os.path.join(self.reports_dir, "m6_skipped_records.jsonl"),
            self.LATEST_POINTER_FILE: os.path.join(self.output_dir, self.LATEST_POINTER_FILE),
        }
        for fname, dst in mapping.items():
            src = os.path.join(backup_dir, fname)
            if os.path.exists(src):
                tmp_dst = dst + ".tmp"
                shutil.copy2(src, tmp_dst)
                os.replace(tmp_dst, dst)
            elif os.path.exists(dst):
                try:
                    os.remove(dst)
                except OSError:
                    pass

    @classmethod
    def resolve_latest_run(
        cls,
        output_dir: str,
        expected_scope: Optional[str] = None,
        allow_mock: bool = False
    ) -> Dict[str, Any]:
        """
        Authoritative resolver for M7 and downstream milestones.
        Reads LATEST.json, verifies the run directory and COMMITTED marker,
        re-verifies cryptographic checksums, and enforces scope and mock invariants.
        Fails closed on uncommitted, incomplete, or mismatched runs.
        """
        pointer_path = os.path.join(output_dir, cls.LATEST_POINTER_FILE)
        if not os.path.exists(pointer_path):
            raise FileNotFoundError(
                f"No authoritative run pointer found at {pointer_path}. "
                "No embedding runs have been committed."
            )

        with open(pointer_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)

        run_rel_dir = pointer.get("run_dir")
        if not run_rel_dir:
            raise ValueError(f"Malformed pointer in {pointer_path}: missing 'run_dir'")

        run_dir = os.path.normpath(os.path.join(output_dir, run_rel_dir))
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Authoritative run directory not found: {run_dir}")

        committed_path = os.path.join(run_dir, cls.COMMIT_MARKER_FILE)
        if not os.path.exists(committed_path):
            raise RuntimeError(
                f"Run directory {run_dir} is uncommitted (missing {cls.COMMIT_MARKER_FILE}). "
                "Refusing to consume incomplete run."
            )

        with open(committed_path, "r", encoding="utf-8") as f:
            commit_data = json.load(f)

        # Enforce scope isolation
        run_scope = commit_data.get("scope")
        if expected_scope is not None and run_scope != expected_scope:
            raise ValueError(
                f"Scope mismatch in run {commit_data.get('run_id')}: "
                f"expected '{expected_scope}', got '{run_scope}'."
            )

        # Enforce mock protection
        is_mock = commit_data.get("is_mock", False)
        if not allow_mock and is_mock:
            raise ValueError(
                f"Refusing to consume mock embeddings in production scope. "
                f"Run {commit_data.get('run_id')} is tagged is_mock=True."
            )

        # Verify artifacts and checksums
        artifacts = commit_data.get("artifacts", {})
        vec_file = artifacts.get("vector_file")
        vec_hash = artifacts.get("vector_sha256")
        manifest_file = artifacts.get("manifest_file")
        manifest_hash = artifacts.get("manifest_sha256")
        report_file = artifacts.get("report_file")
        report_hash = artifacts.get("report_sha256")
        skipped_file = artifacts.get("skipped_file")

        if not vec_file or not manifest_file or not report_file:
            raise ValueError(f"Incomplete artifact manifest in {committed_path}")

        vec_full_path = os.path.join(run_dir, vec_file)
        manifest_full_path = os.path.join(run_dir, manifest_file)
        report_full_path = os.path.join(run_dir, report_file)
        skipped_full_path = os.path.join(run_dir, skipped_file) if skipped_file else None

        for path, expected_hash, name in [
            (vec_full_path, vec_hash, "vector file"),
            (manifest_full_path, manifest_hash, "manifest file"),
            (report_full_path, report_hash, "report file"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing committed artifact: {name} at {path}")
            with open(path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Integrity corruption detected in committed run {commit_data.get('run_id')}: "
                    f"{name} SHA-256 mismatch. Expected {expected_hash}, got {actual_hash}."
                )

        if skipped_full_path and not os.path.exists(skipped_full_path):
            raise FileNotFoundError(f"Missing committed skipped-records artifact at {skipped_full_path}")

        return {
            "run_id": commit_data.get("run_id"),
            "run_dir": run_dir,
            "scope": run_scope,
            "is_mock": is_mock,
            "model_identifier": commit_data.get("model_identifier"),
            "vector_count": commit_data.get("vector_count"),
            "manifest_count": commit_data.get("manifest_count"),
            "matrix_shape": commit_data.get("matrix_shape"),
            "vector_file": vec_full_path,
            "manifest_file": manifest_full_path,
            "report_file": report_full_path,
            "skipped_file": skipped_full_path,
            "committed_at": commit_data.get("committed_at"),
            "artifacts": artifacts
        }
