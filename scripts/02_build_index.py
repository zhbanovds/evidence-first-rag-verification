import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR
from src.indexing.chroma_store import build_chroma_index


def load_chunks(mode: str) -> list[dict]:
    chunks_path = PROCESSED_DIR / f"{mode}_chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. Run scripts/01_prepare_data.py first."
        )

    with chunks_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index for RAG chunks.")
    parser.add_argument("--mode", choices=["clean", "conflict"], required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the embedding model into the project cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[build_index] Loading chunks for mode: {args.mode}")

    chunks = load_chunks(args.mode)
    print(f"[build_index] Chunks loaded: {len(chunks)}")

    collection = build_chroma_index(
        chunks,
        args.mode,
        local_files_only=not args.allow_download,
    )
    print(f"[build_index] Chunks added to Chroma: {collection.count()}")


if __name__ == "__main__":
    main()
