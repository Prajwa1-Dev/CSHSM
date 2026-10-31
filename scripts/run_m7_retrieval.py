import os
import sys
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.embedder import MockEmbedder, BGE_M3_Embedder
from src.retrieval.retriever import CSHSMRetriever

def parse_args():
    parser = argparse.ArgumentParser(
        description="M7 Counter-Speech Knowledge Base: Vector Retrieval Pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Input query text (hate speech comment) to retrieve matching counter-speech for."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest-neighbor candidates to retrieve. Default: 3."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter (e.g. 'Religion', 'Caste', 'Gender')."
    )
    parser.add_argument(
        "--scope",
        choices=["development_provisional", "approved_kb"],
        default="development_provisional",
        help="Embedding scope to retrieve from. Default: 'development_provisional'."
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        default=True,
        help="Allow retrieval over mock development embeddings."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockEmbedder to compute query vector for testing without model weights."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/counter_speech_kb/embeddings",
        help="Directory where committed embedding runs and LATEST.json reside."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 76)
    print("CSHSM — M7 Top-K Counter-Speech Retrieval")
    print("=" * 76)
    print(f"  Query:          {args.query}")
    print(f"  Top-K:          {args.top_k}")
    print(f"  Category:       {args.category or 'All'}")
    print(f"  Scope:          {args.scope}")
    print(f"  Allow Mock:     {args.allow_mock}")
    print(f"  Embedder Mode:  {'MOCK (Test Only)' if args.mock else 'REAL (BAAI/bge-m3)'}")
    print("=" * 76)

    # Configure query embedder
    if args.mock:
        embedder = MockEmbedder(dim=1024)
    else:
        try:
            embedder = BGE_M3_Embedder(model_name="BAAI/bge-m3", device="cpu")
        except ImportError as e:
            print(f"\nERROR: Missing dependencies for real BGE-M3 text query encoding:\n  {e}", file=sys.stderr)
            print("To run with mock embeddings in test environment, use: --mock", file=sys.stderr)
            sys.exit(1)

    try:
        retriever = CSHSMRetriever.from_committed_run(
            output_dir=args.output_dir,
            scope=args.scope,
            allow_mock=args.allow_mock,
            embedder=embedder
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize CSHSMRetriever from committed run:\n  {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = retriever.retrieve_by_text(
            query_text=args.query,
            top_k=args.top_k,
            category_filter=args.category
        )

        print(f"\n[M7] Retrieval Status:  {result.retrieval_status.upper()}")
        print(f"[M7] Backend Engine:    {result.backend}")
        print(f"[M7] Source Run ID:     {result.run_id}")
        print(f"[M7] Scope / Approval:  {result.scope} ({result.approval_status})")
        print(f"[M7] Notice:            {result.disclaimer}")
        print("-" * 76)

        if not result.matches:
            print("No matching candidates found.")
        else:
            for m in result.matches:
                print(f"  Rank {m.rank} | Score: {m.score:.4f} | KB ID: {m.kb_id} | Category: {m.category}")
                print(f"    Hate Text:      {m.hate_text}")
                print(f"    Counter-Speech: {m.counter_speech}")
                print(f"    Quality/Review: {m.quality_status} / {m.review_status}")
                print()

        print("=" * 76)
        print("[M7] Note: Threshold calibration and routing are deferred to M8.")
        print("=" * 76)

    except Exception as e:
        print(f"\nERROR during retrieval: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
