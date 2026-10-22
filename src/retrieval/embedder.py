import os
import re
import json
import math
import array
import struct
import shutil
import hashlib
import gc
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
        np.save(filepath, arr)
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
        device: str = "cpu",
        batch_size: int = 4,
        max_length: int = 512,
        revision: Optional[str] = None
    ):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._revision = revision
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

        kwargs = {"device": self._device}
        if self._revision:
            kwargs["revision"] = self._revision
        self._model = SentenceTransformer(self._model_name, **kwargs)
        self._model.max_seq_length = self._max_length

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        import torch
        with torch.no_grad():
            outputs = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            gc.collect()
            return outputs.tolist()


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
                if e.quality_status != "approved" or e.review_status != "approved":
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
                    entry = KBEntry(**record)
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

        # 7. Atomic persistence & backup
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # Backup prior artifacts if existing
        self._backup_existing_artifacts()

        # Use distinct filenames for mock vs real embeddings to prevent accidental acceptance
        prefix = "bge_m3_mock" if self.embedder.is_mock else "bge_m3"
        vec_path = os.path.join(self.output_dir, f"{prefix}_dense.npy")
        manifest_path = os.path.join(self.output_dir, f"{prefix}_manifest.jsonl")
        report_path = os.path.join(self.reports_dir, "m6_embedding_report.json")
        skipped_path = os.path.join(self.reports_dir, "m6_skipped_records.jsonl")

        tmp_vec_path = vec_path + ".tmp"
        tmp_manifest_path = manifest_path + ".tmp"
        tmp_report_path = report_path + ".tmp"

        try:
            # Write vector matrix
            save_npy_f32(tmp_vec_path, all_embeddings, matrix_shape)

            # Write manifest
            with open(tmp_manifest_path, "w", encoding="utf-8") as f:
                for rec in manifest_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Write skipped records if any
            if skipped_records:
                with open(skipped_path, "w", encoding="utf-8") as f:
                    for s in skipped_records:
                        f.write(json.dumps(s) + "\n")

            # Compute output hashes
            with open(tmp_vec_path, "rb") as f:
                vec_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(tmp_manifest_path, "rb") as f:
                manifest_sha256 = hashlib.sha256(f.read()).hexdigest()

            # Build metadata report
            now_iso = datetime.now(timezone.utc).isoformat()
            scope_disclaimer = (
                "PROVISIONAL CANDIDATES — Embedded for retrieval development only; counter-speech remains unapproved by M5.4."
                if self.scope == "development_provisional" else
                "APPROVED KB — Formally evaluated and approved counter-speech knowledge base."
            )
            mock_note = "MOCK EMBEDDINGS — Strictly for tests and engineering validation." if self.embedder.is_mock else None

            report = {
                "timestamp": now_iso,
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
                "artifacts": {
                    "vector_file": os.path.basename(vec_path),
                    "vector_sha256": vec_sha256,
                    "manifest_file": os.path.basename(manifest_path),
                    "manifest_sha256": manifest_sha256
                },
                "disclaimer": scope_disclaimer,
                "mock_notice": mock_note
            }

            with open(tmp_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            # Atomic rename
            os.replace(tmp_vec_path, vec_path)
            os.replace(tmp_manifest_path, manifest_path)
            os.replace(tmp_report_path, report_path)

        except Exception as e:
            # Clean up temporary files on error
            for p in [tmp_vec_path, tmp_manifest_path, tmp_report_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            raise RuntimeError(f"Embedding pipeline failed and aborted safely: {e}") from e

        return report

    def _backup_existing_artifacts(self):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.reports_dir, "backups", f"m6_pre_run_{timestamp}")

        prefix = "bge_m3_mock" if self.embedder.is_mock else "bge_m3"
        files = [
            os.path.join(self.output_dir, f"{prefix}_dense.npy"),
            os.path.join(self.output_dir, f"{prefix}_manifest.jsonl"),
            os.path.join(self.reports_dir, "m6_embedding_report.json"),
        ]
        backed_up = False
        for fpath in files:
            if os.path.exists(fpath):
                os.makedirs(backup_dir, exist_ok=True)
                dest = os.path.join(backup_dir, os.path.basename(fpath))
                shutil.copy2(fpath, dest)
                backed_up = True
        return backup_dir if backed_up else None
