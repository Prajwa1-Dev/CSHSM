"""
Evaluate BGE-M3 Dense Retrieval across all six categories with paraphrased queries.
Measures cosine similarity, rank-1 accuracy, latency, and GPU VRAM usage.
Compares BGE-M3 results with MockEmbedder baseline.
"""

import os
import sys
import json
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.embedder import BGE_M3_Embedder, MockEmbedder
from src.retrieval.retriever import CSHSMRetriever


PARAPHRASE_TEST_CASES = [
    {
        "category": "Gender",
        "target_kb_id": "DEMO-KB-00001",
        "original": "Women do not belong in leadership or engineering, their brains are built only for housework.",
        "paraphrase": "Females are incapable of technical leadership or technical engineering roles."
    },
    {
        "category": "Political",
        "target_kb_id": "DEMO-KB-00005",
        "original": "Anyone voting for the opposition party is a traitor and should lose their right to citizenship.",
        "paraphrase": "Citizens who cast their vote for opposition politicians are national traitors."
    },
    {
        "category": "Religion",
        "target_kb_id": "DEMO-KB-00009",
        "original": "All Muslims are violent radicals who are secretly planning to wage war on our homeland.",
        "paraphrase": "Muslim communities are extremists secretly plotting warfare against our nation."
    },
    {
        "category": "Geo-political",
        "target_kb_id": "DEMO-KB-00015",
        "original": "Foreign workers are stealing jobs from native citizens and should be rounded up and deported immediately.",
        "paraphrase": "Migrant workers are taking local employment and should all be immediately deported."
    },
    {
        "category": "Violence",
        "target_kb_id": "DEMO-KB-00018",
        "original": "We must form armed vigilante mobs and lynch these criminals publicly instead of relying on courts.",
        "paraphrase": "Citizens must organize vigilante groups to execute suspects in the street rather than waiting for judges."
    },
    {
        "category": "Others",
        "target_kb_id": "DEMO-KB-00022",
        "original": "People with cognitive and physical disabilities are useless burdens consuming resources meant for productive citizens.",
        "paraphrase": "Disabled individuals are just unproductive burdens draining public welfare."
    }
]


def run_evaluation():
    print("=" * 80)
    print("CSHSM PHASE 4A: SEMANTIC RETRIEVAL BENCHMARK ACROSS 6 CATEGORIES")
    print("=" * 80)

    # 1. Initialize BGE-M3 Retriever
    embedder_bge = BGE_M3_Embedder(device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=True)
    retriever_bge = CSHSMRetriever.from_committed_run(
        output_dir="data/counter_speech_kb/embeddings",
        allow_mock=False,
        embedder=embedder_bge,
        force_backend="exact_ip_fallback"
    )

    # 2. Initialize MockEmbedder baseline
    embedder_mock = MockEmbedder(dim=1024)

    # Warmup BGE-M3
    _ = retriever_bge.retrieve_by_text("Warmup query", top_k=1)

    results = []

    print(f"\n{'Category':<14} | {'Target KB':<13} | {'Matched KB':<13} | {'BGE-M3 Score':<12} | {'Category Match?':<15} | {'Latency (ms)':<12}")
    print("-" * 85)

    for tc in PARAPHRASE_TEST_CASES:
        category = tc["category"]
        target_id = tc["target_kb_id"]
        query = tc["paraphrase"]

        # Track VRAM before query
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_start = torch.cuda.memory_allocated() / (1024 ** 2)

        # Measure BGE-M3 Retrieval latency
        t0 = time.perf_counter()
        bge_res = retriever_bge.retrieve_by_text(query, top_k=3)
        t_lat_ms = (time.perf_counter() - t0) * 1000

        # VRAM peak during query
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        top_match = bge_res.matches[0] if bge_res.matches else None
        top_id = top_match.kb_id if top_match else "NONE"
        top_score = top_match.score if top_match else 0.0
        top_cat = top_match.category if top_match else "NONE"
        cat_match = (top_cat == category)

        print(f"{category:<14} | {target_id:<13} | {top_id:<13} | {top_score:<12.4f} | {str(cat_match):<15} | {t_lat_ms:<12.1f}")

        # Also compute MockEmbedder baseline score on this query vs target KB text
        mock_q_vec = embedder_mock.embed_batch([query])[0]
        mock_t_vec = embedder_mock.embed_batch([tc["original"]])[0]
        mock_score = sum(q * t for q, t in zip(mock_q_vec, mock_t_vec))

        results.append({
            "category": category,
            "target_id": target_id,
            "original_hate_text": tc["original"],
            "paraphrase_query": query,
            "bge_top_id": top_id,
            "bge_top_category": top_cat,
            "bge_top_score": round(top_score, 4),
            "category_match": cat_match,
            "exact_id_match": (top_id == target_id),
            "latency_ms": round(t_lat_ms, 2),
            "vram_peak_mb": round(vram_peak, 2),
            "mock_baseline_score": round(mock_score, 4),
            "all_bge_matches": [
                {
                    "rank": m.rank,
                    "kb_id": m.kb_id,
                    "score": round(m.score, 4),
                    "category": m.category,
                    "hate_text": m.hate_text
                }
                for m in bge_res.matches
            ]
        })

    # Summary Statistics
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_score = sum(r["bge_top_score"] for r in results) / len(results)
    avg_mock_score = sum(r["mock_baseline_score"] for r in results) / len(results)
    cat_accuracy = sum(1 for r in results if r["category_match"]) / len(results) * 100
    exact_accuracy = sum(1 for r in results if r["exact_id_match"]) / len(results) * 100

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY & METRICS")
    print("=" * 80)
    print(f"Total Test Cases:            {len(results)}")
    print(f"Category Match Accuracy:     {cat_accuracy:.1f}% ({sum(1 for r in results if r['category_match'])}/{len(results)})")
    print(f"Exact Target KB Accuracy:    {exact_accuracy:.1f}% ({sum(1 for r in results if r['exact_id_match'])}/{len(results)})")
    print(f"Average BGE-M3 Paraphrase Similarity: {avg_score:.4f}")
    print(f"Average Mock Baseline Similarity:      {avg_mock_score:.4f}")
    print(f"Average Query Latency:       {avg_latency:.2f} ms")
    if torch.cuda.is_available():
        print(f"Peak VRAM during retrieval:  {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

    # Write report
    report_path = "reports/phase4a_bge_m3_retrieval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "BAAI/bge-m3",
            "backend": "exact_ip_fallback",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "total_test_cases": len(results),
            "category_match_accuracy_percent": cat_accuracy,
            "exact_kb_match_accuracy_percent": exact_accuracy,
            "average_bge_similarity": round(avg_score, 4),
            "average_mock_similarity": round(avg_mock_score, 4),
            "average_latency_ms": round(avg_latency, 2),
            "test_results": results
        }, f, indent=2)

    print(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    run_evaluation()
