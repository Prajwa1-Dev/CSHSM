"""
Direct mathematical & performance comparison: FAISS IndexFlatIP vs ExactInnerProductIndex.
Evaluates ranking equivalence, numerical tolerance, and search latency.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.embedder import BGE_M3_Embedder, load_npy_f32
from src.retrieval.retriever import FAISSVectorIndex, ExactInnerProductIndex, CSHSMRetriever


def main():
    print("=" * 80)
    print("CSHSM PHASE 4B: FAISS IndexFlatIP vs ExactInnerProductIndex COMPARISON")
    print("=" * 80)

    # 1. Load committed vectors
    vec_path = "data/counter_speech_kb/embeddings/bge_m3_dense.npy"
    manifest_path = "data/counter_speech_kb/embeddings/bge_m3_manifest.jsonl"
    vectors, shape = load_npy_f32(vec_path)
    n_rows, dim = shape
    print(f"Loaded {n_rows} vectors of dimension {dim} from {vec_path}")

    # 2. Populate both indices
    faiss_idx = FAISSVectorIndex(dim=dim)
    exact_idx = ExactInnerProductIndex(dim=dim)

    t0 = time.perf_counter()
    faiss_idx.add(vectors)
    t_add_faiss = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    exact_idx.add(vectors)
    t_add_exact = (time.perf_counter() - t0) * 1000

    print(f"Index Construction Time: FAISS={t_add_faiss:.3f} ms, Exact={t_add_exact:.3f} ms")
    assert faiss_idx.ntotal == exact_idx.ntotal == n_rows

    # 3. Test queries
    test_queries = [
        "Females are incapable of technical leadership or engineering roles.",
        "Citizens who cast their vote for opposition politicians are national traitors.",
        "Muslim communities are extremists secretly plotting warfare against our nation.",
        "Migrant workers are taking local employment and should all be immediately deported.",
        "Citizens must organize vigilante groups to execute suspects in the street rather than waiting for judges.",
        "Disabled individuals are just unproductive burdens draining public welfare."
    ]

    embedder = BGE_M3_Embedder(device="cuda", local_files_only=True)
    query_vectors = embedder.embed_batch(test_queries)

    print(f"\n{'Query':<35} | {'FAISS Top-1 (Score)':<20} | {'Exact Top-1 (Score)':<20} | {'Max Delta':<10} | {'Ranking Match?'}")
    print("-" * 105)

    max_score_diff = 0.0
    all_ranks_matched = True
    faiss_times = []
    exact_times = []

    for idx, (query, q_vec) in enumerate(zip(test_queries, query_vectors)):
        # Search FAISS
        t0 = time.perf_counter()
        f_scores, f_indices = faiss_idx.search([q_vec], k=3)
        faiss_times.append((time.perf_counter() - t0) * 1000)

        # Search Exact
        t0 = time.perf_counter()
        e_scores, e_indices = exact_idx.search([q_vec], k=3)
        exact_times.append((time.perf_counter() - t0) * 1000)

        f_top = f"{f_indices[0][0]} ({f_scores[0][0]:.4f})"
        e_top = f"{e_indices[0][0]} ({e_scores[0][0]:.4f})"

        rank_match = (f_indices[0] == e_indices[0])
        if not rank_match:
            all_ranks_matched = False

        diffs = [abs(fs - es) for fs, es in zip(f_scores[0], e_scores[0])]
        query_max_diff = max(diffs)
        if query_max_diff > max_score_diff:
            max_score_diff = query_max_diff

        print(f"{query[:35]:<35} | {f_top:<20} | {e_top:<20} | {query_max_diff:<10.6f} | {str(rank_match)}")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"All Top-3 Rankings Identical: {all_ranks_matched}")
    print(f"Maximum Float32 Score Delta:  {max_score_diff:.8f} (within 1e-5 numerical tolerance)")
    print(f"Average FAISS Search Latency: {sum(faiss_times) / len(faiss_times):.4f} ms")
    print(f"Average Exact Search Latency: {sum(exact_times) / len(exact_times):.4f} ms")

    assert all_ranks_matched, "Mismatch in top-K rankings!"
    assert max_score_diff < 1e-5, f"Score delta too high: {max_score_diff}"
    print("\nMathematical equivalence between FAISS IndexFlatIP and ExactInnerProductIndex VERIFIED.")


if __name__ == "__main__":
    main()
