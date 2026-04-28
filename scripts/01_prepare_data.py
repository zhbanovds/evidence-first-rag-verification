import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR
from src.data.chunker import chunk_documents
from src.data.loader import load_documents


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_mode(mode: str) -> None:
    print(f"[prepare_data] Processing mode: {mode}")

    documents = load_documents(mode)
    chunks = chunk_documents(documents)

    documents_path = PROCESSED_DIR / f"{mode}_documents.jsonl"
    chunks_path = PROCESSED_DIR / f"{mode}_chunks.jsonl"

    write_jsonl(documents_path, documents)
    write_jsonl(chunks_path, chunks)

    print(f"[prepare_data] Documents read: {len(documents)}")
    print(f"[prepare_data] Chunks created: {len(chunks)}")
    print(f"[prepare_data] Saved documents: {documents_path}")
    print(f"[prepare_data] Saved chunks: {chunks_path}")


def main() -> None:
    processed_modes = ["clean", "conflict"]

    for mode in processed_modes:
        prepare_mode(mode)

    print(f"[prepare_data] Processed modes: {', '.join(processed_modes)}")


if __name__ == "__main__":
    main()
