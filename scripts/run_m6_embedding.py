import os
import sys
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.embedder import (
    BaseEmbedder,
    MockEmbedder,
    BGE_M3_Embedder,
    EmbeddingPipeline
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="M6 Counter-Speech Knowledge Base: Dense Embedding Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/counter_speech_kb/filtered_candidates.jsonl",
        help="Path to input KB JSONL file."
    )
    parser.add_argument(
        "--scope",
        choices=["development_provisional", "approved_kb"],
        default="development_provisional",
        help="Embedding scope. 'approved_kb' fails closed if any candidate is unapproved. 'development_provisional' embeds unapproved candidates explicitly marked for dev/prototyping."
    )
    parser.add_argument(
        "--strategy",
        choices=["hate_text", "structured_category_hate"],
        default="hate_text",
        help="Embedding text strategy. Default: 'hate_text'. Optional: 'structured_category_hate'."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size. Default 4 for conservative memory footprint on 8GB RAM."
    )
    parser.add_argument(
        "--dev-limit",
        type=int,
        default=100,
        help="Subset size limit for development smoke test. Default 100."
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Process the complete input dataset, overriding --dev-limit."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic MockEmbedder for engineering tests without downloading weights."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for embedding inference ('cpu')."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    limit = None if args.full_dataset else args.dev_limit
    print("=" * 76)
    print("CSHSM — M6 BGE-M3 Dense Embedding Pipeline")
    print("=" * 76)
    print(f"  Input File:     {args.input}")
    print(f"  Scope:          {args.scope}")
    print(f"  Strategy:       {args.strategy}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Record Limit:   {'Full Dataset' if limit is None else str(limit) + ' (dev subset)'}")
    print(f"  Embedder Mode:  {'MOCK (Test Only)' if args.mock else 'REAL (BAAI/bge-m3)'}")
    print("=" * 76)

    if args.mock:
        embedder: BaseEmbedder = MockEmbedder(dim=1024)
    else:
        try:
            embedder = BGE_M3_Embedder(
                model_name="BAAI/bge-m3",
                device=args.device,
                batch_size=args.batch_size,
                max_length=512
            )
        except ImportError as e:
            print(f"\nERROR: Missing dependencies for real BGE-M3 inference:\n  {e}", file=sys.stderr)
            sys.exit(1)

    pipeline = EmbeddingPipeline(
        embedder=embedder,
        scope=args.scope,
        strategy=args.strategy,
        batch_size=args.batch_size,
        dev_limit=limit
    )

    try:
        report = pipeline.run(args.input)
        print("\n[M6] Embedding pipeline completed successfully.")
        print(f"[M6] Model:        {report['model_identifier']}")
        print(f"[M6] Is Mock:      {report['is_mock']}")
        print(f"[M6] Scope:        {report['kb_scope']} (Approval status: {report['approval_status']})")
        print(f"[M6] Shape:        {report['matrix_shape']}")
        print(f"[M6] Vector File:  {report['artifacts']['vector_file']}")
        print(f"[M6] Manifest:     {report['artifacts']['manifest_file']}")
        print(f"[M6] Report Saved: reports/m6_embedding_report.json")
    except Exception as e:
        print(f"\nERROR in M6 pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
