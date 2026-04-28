import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOP_K
from src.retrieval.hybrid_retriever import search_hybrid
from src.retrieval.vector_retriever import search_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual retrieval check.")
    parser.add_argument("--mode", choices=["clean", "conflict"], required=True)
    parser.add_argument("--retriever", choices=["vector", "hybrid"], default="hybrid")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def format_text(text: str, max_chars: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def main() -> None:
    args = parse_args()
    if args.retriever == "vector":
        chunks = search_vector(args.query, mode=args.mode, top_k=args.top_k)
    else:
        chunks = search_hybrid(args.query, mode=args.mode, top_k=args.top_k)

    print(f"[test_retrieval] mode: {args.mode}")
    print(f"[test_retrieval] retriever: {args.retriever}")
    print(f"[test_retrieval] query: {args.query}")
    print(f"[test_retrieval] top_k: {args.top_k}")
    print()

    for rank, chunk in enumerate(chunks, start=1):
        score = chunk["score"]
        score_text = f"{score:.4f}" if score is not None else "n/a"

        print(f"Rank: {rank}")
        print(f"Score: {score_text}")
        print(f"doc_id: {chunk['doc_id']}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source_path: {chunk['source_path']}")
        print(f"is_conflict_source: {chunk['is_conflict_source']}")
        print("text:")
        print(format_text(chunk["text"]))
        print("-" * 80)


if __name__ == "__main__":
    main()
