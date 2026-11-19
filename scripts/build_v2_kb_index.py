"""
Build and index the 300-entry Romanized Kannada-English Knowledge Base V2 using BGE-M3.

This script:
1. Reads the canonical V2 dataset from data/counter_speech_kb/cshsm_kannada_english_kb_v2.jsonl.
2. Uses local BGE-M3 model (offline, CUDA if available) to compute 1024-dimensional normalized dense vectors.
3. Commits the run artifacts into isolated data/counter_speech_kb/embeddings_v2/ directory.
4. Validates the 300 x 1024 vector matrix, unit L2 norms (1.0 +/- 1e-4), and manifest alignment.
5. Verifies CSHSMRetriever instantiation and test retrieval.
"""

import os
import sys
import json
import time
import math
from typing import Dict, Any

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.retrieval.embedder import BGE_M3_Embedder, EmbeddingPipeline, load_npy_f32
from src.retrieval.retriever import CSHSMRetriever


def main():
    print("=" * 80)
    print("CSHSM: KNOWLEDGE BASE V2 EMBEDDING & INDEX CONSTRUCTION (BGE-M3)")
    print("=" * 80)

    kb_input_path = "data/counter_speech_kb/cshsm_kannada_english_kb_v2.jsonl"
    output_dir = "data/counter_speech_kb/embeddings_v2"
    reports_dir = "reports"

    if not os.path.exists(kb_input_path):
        raise FileNotFoundError(f"V2 KB dataset not found at {kb_input_path}")

    # Check CUDA availability and baseline VRAM
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_start_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_start_res = torch.cuda.memory_reserved() / (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA ({device_name})")
        print(f"Baseline VRAM: {vram_start_alloc:.2f} MB allocated, {vram_start_res:.2f} MB reserved")
    else:
        print("Device: CPU (CUDA not available)")

    t0 = time.perf_counter()

    # Step 1: Initialize BGE_M3_Embedder in offline mode
    print("\n[1/5] Initializing BGE_M3_Embedder (local_files_only=True)...")
    t_load_start = time.perf_counter()
    embedder = BGE_M3_Embedder(
        model_name="BAAI/bge-m3",
        device="cuda" if cuda_available else "cpu",
        batch_size=16,
        local_files_only=True
    )
    t_load_end = time.perf_counter()
    print(f"      Model loaded in {t_load_end - t_load_start:.2f} s")
    print(f"      Model identifier: {embedder.model_identifier}")
    print(f"      Is mock:          {embedder.is_mock}")
    print(f"      Dimension:        {embedder.dimension}")
    print(f"      Device:           {embedder.device}")

    # Step 2: Set up EmbeddingPipeline with isolated output_dir
    print(f"\n[2/5] Setting up EmbeddingPipeline (output_dir='{output_dir}')...")
    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope="development_provisional",
        strategy="hate_text",
        batch_size=16,
        output_dir=output_dir,
        reports_dir=reports_dir
    )

    verify_only = "--verify-only" in sys.argv
    latest_pointer = os.path.join(output_dir, "LATEST.json")

    # Step 3: Run embedding pipeline on V2 KB if not verify-only
    if verify_only and os.path.exists(latest_pointer):
        print(f"\n[3/5] Skipping embedding generation (--verify-only specified). Loading LATEST.json...")
        with open(latest_pointer, "r", encoding="utf-8") as f:
            latest_info = json.load(f)
        report = {
            "run_id": latest_info["run_id"],
            "run_dir": latest_info["run_dir"],
            "embedded_record_count": latest_info["vector_count"],
            "matrix_shape": latest_info["matrix_shape"],
        }
        print(f"      Run ID:           {report['run_id']}")
        print(f"      Run directory:    {report['run_dir']}")
        print(f"      Embedded count:   {report['embedded_record_count']}")
        print(f"      Matrix shape:     {report['matrix_shape']}")
    else:
        print(f"\n[3/5] Generating BGE-M3 embeddings for 300 V2 entries...")
        t_emb_start = time.perf_counter()
        report = pipeline.run(kb_input_path)
        t_emb_end = time.perf_counter()
        emb_duration = t_emb_end - t_emb_start

        print(f"      Run ID:           {report['run_id']}")
        print(f"      Run directory:    {report['run_dir']}")
        print(f"      Embedded count:   {report['embedded_record_count']}")
        print(f"      Matrix shape:     {report['matrix_shape']}")
        print(f"      Elapsed time:     {emb_duration:.2f} s")
        print(f"      Throughput:       {report['embedded_record_count'] / emb_duration:.2f} entries/s")

    if cuda_available:
        vram_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram_peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"      Peak VRAM allocated: {vram_peak_alloc:.2f} MB")
        print(f"      Peak VRAM reserved:  {vram_peak_res:.2f} MB")

    # Step 4: Validate committed artifacts on disk
    print("\n[4/5] Validating committed artifacts...")
    run_dir = os.path.join(output_dir, report["run_dir"])
    vec_path = os.path.join(run_dir, "bge_m3_dense.npy")
    manifest_path = os.path.join(run_dir, "bge_m3_manifest.jsonl")
    commit_marker = os.path.join(run_dir, "COMMITTED")
    latest_pointer = os.path.join(output_dir, "LATEST.json")

    assert os.path.exists(vec_path), f"Vectors missing: {vec_path}"
    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    assert os.path.exists(commit_marker), f"COMMITTED marker missing: {commit_marker}"
    assert os.path.exists(latest_pointer), f"LATEST.json missing: {latest_pointer}"

    # Verify vector matrix shape and norms
    vectors, shape = load_npy_f32(vec_path)
    print(f"      Loaded vector matrix: {shape[0]} rows x {shape[1]} columns")
    assert shape == (300, 1024), f"Expected shape (300, 1024), got {shape}"

    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    min_norm, max_norm = min(norms), max(norms)
    print(f"      Min L2 norm: {min_norm:.6f}, Max L2 norm: {max_norm:.6f}")
    assert all(abs(n - 1.0) < 1e-4 for n in norms), f"L2 norm check failed! Outside [0.9999, 1.0001]"
    print("      -> Unit L2 normalization verified for all 300 vectors.")

    # Verify manifest records
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_records = [json.loads(line) for line in f if line.strip()]
    assert len(manifest_records) == 300, f"Expected 300 manifest records, got {len(manifest_records)}"

    cat_counts = {}
    for r in manifest_records:
        cat = r.get("category", "UNKNOWN")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        assert r.get("quality_status") == "pending_review", f"Expected quality_status='pending_review', got {r.get('quality_status')}"
        assert r.get("scope") == "development_provisional", f"Expected scope='development_provisional', got {r.get('scope')}"

    print(f"      Category distribution in manifest: {cat_counts}")
    assert len(cat_counts) == 6, f"Expected 6 categories, got {len(cat_counts)}"
    for cat, count in cat_counts.items():
        assert count == 50, f"Expected 50 entries for {cat}, got {count}"
    print("      -> All 6 categories evenly balanced at 50 records each.")

    # Step 5: Test CSHSMRetriever instantiation from committed V2 run
    print("\n[5/5] Testing CSHSMRetriever instantiation and test retrieval...")
    retriever = CSHSMRetriever.from_committed_run(
        output_dir=output_dir,
        allow_mock=False,
        embedder=embedder
    )
    print(f"      Retriever backend:   {retriever.index.backend_name}")
    print(f"      Total indexed:       {retriever.index.ntotal}")
    print(f"      Retriever is_mock:   {retriever.run_metadata['is_mock']}")
    print(f"      Model identifier:    {retriever.run_metadata['model_identifier']}")

    # Test retrieval with a sample query from V2 dataset
    test_query = "Hudugir ge driving baralla road alli bandre accident guarantee"
    res = retriever.retrieve_by_text(test_query, top_k=3)
    matches = res.matches
    print(f"\n      Test query: '{test_query}'")
    print(f"      Top-1 retrieved ID:       {matches[0].kb_id}")
    print(f"      Top-1 category:           {matches[0].category}")
    print(f"      Top-1 similarity score:   {matches[0].score:.4f}")
    print(f"      Top-1 counter-speech:     {matches[0].counter_speech[:80]}...")

    assert matches[0].kb_id == "CSHSM-KN-00001", f"Expected top-1 CSHSM-KN-00001, got {matches[0].kb_id}"
    assert matches[0].category == "Gender", f"Expected Category Gender, got {matches[0].category}"
    assert matches[0].score > 0.90, f"Expected high similarity >0.90 for near-verbatim query, got {matches[0].score}"

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"KB V2 Index Construction and Verification COMPLETED SUCCESSFULLY in {total_time:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
