"""
Re-embed the 24 demonstration Knowledge Base entries using real BGE-M3 dense embeddings.
Records execution time, latency per batch, GPU memory usage, and verifies index creation.
"""

import os
import sys
import json
import time
import math
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.embedder import BGE_M3_Embedder, EmbeddingPipeline, load_npy_f32
from src.retrieval.retriever import CSHSMRetriever, ExactInnerProductIndex, create_vector_index


def main():
    print("=" * 80)
    print("CSHSM PHASE 4A: REAL BGE-M3 EMBEDDING OF DEMONSTRATION KNOWLEDGE BASE")
    print("=" * 80)

    kb_input_path = "data/counter_speech_kb/demo_approved_kb.jsonl"
    if not os.path.exists(kb_input_path):
        raise FileNotFoundError(f"Demo KB file not found at {kb_input_path}")

    # Check CUDA memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_start_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_start_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA ({device_name})")
        print(f"VRAM baseline: {vram_start_allocated:.2f} MB allocated, {vram_start_reserved:.2f} MB reserved")
    else:
        device_name = "CPU"
        print("Device: CPU (CUDA not available)")

    t0 = time.perf_counter()
    print("\n[1/4] Initializing BGE_M3_Embedder on CUDA (offline mode)...")
    embedder = BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        local_files_only=True
    )
    print(f"      Model identifier: {embedder.model_identifier}")
    print(f"      Embedder is_mock: {embedder.is_mock}")
    print(f"      Target dimension: {embedder.dimension}")
    print(f"      Embedder device:  {embedder.device}")

    print("\n[2/4] Setting up EmbeddingPipeline and running on demo KB...")
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="development_provisional",
        batch_size=8,
        output_dir="data/counter_speech_kb/embeddings",
        reports_dir="reports"
    )

    t_emb_start = time.perf_counter()
    report = pipeline.run(kb_input_path)
    t_emb_end = time.perf_counter()
    embedding_duration = t_emb_end - t_emb_start

    print(f"      Run ID: {report['run_id']}")
    print(f"      Embedded records: {report['embedded_record_count']}")
    print(f"      Matrix shape: {report['matrix_shape']}")
    print(f"      Embedding duration: {embedding_duration:.3f} s")
    print(f"      Throughput: {report['embedded_record_count'] / embedding_duration:.2f} entries/sec")

    # VRAM stats after embedding
    if torch.cuda.is_available():
        vram_peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"      Peak VRAM allocated: {vram_peak_allocated:.2f} MB")
        print(f"      Peak VRAM reserved:  {vram_peak_reserved:.2f} MB")

    print("\n[3/4] Verifying committed artifacts on disk...")
    run_dir = os.path.join(pipeline.output_dir, report["run_dir"])
    vec_path = os.path.join(run_dir, "bge_m3_dense.npy")
    manifest_path = os.path.join(run_dir, "bge_m3_manifest.jsonl")

    assert os.path.exists(vec_path), f"Vector file missing: {vec_path}"
    assert os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}"

    vectors, shape = load_npy_f32(vec_path)
    print(f"      Loaded vector matrix: {shape[0]} rows x {shape[1]} columns")
    assert shape == (24, 1024), f"Unexpected shape {shape}"

    # Verify L2 norms
    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    print(f"      Min L2 norm: {min(norms):.6f}, Max L2 norm: {max(norms):.6f}")
    assert all(abs(n - 1.0) < 1e-4 for n in norms), "Norm check failed!"

    print("\n[4/4] Verifying CSHSMRetriever instantiation from committed run with ExactInnerProductIndex...")
    retriever = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=embedder,
        force_backend="exact_ip_fallback"
    )
    print(f"      Retriever backend: {retriever.index.backend_name}")
    print(f"      Indexed vectors: {retriever.index.ntotal}")
    print(f"      Retriever is_mock: {retriever.run_metadata['is_mock']}")
    print(f"      Retriever model_id: {retriever.run_metadata['model_identifier']}")

    total_time = time.perf_counter() - t0
    print(f"\nPhase 4A KB Re-embedding completed successfully in {total_time:.2f} s")


if __name__ == "__main__":
    main()
