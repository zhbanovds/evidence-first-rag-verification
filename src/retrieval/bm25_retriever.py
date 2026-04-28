import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.config import BM25_TOP_N, PROCESSED_DIR


TOKEN_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    cleaned = TOKEN_RE.sub(" ", text.lower())
    return [token for token in cleaned.split() if token]


def _chunks_path(mode: str):
    if mode not in {"clean", "conflict"}:
        raise ValueError("mode must be 'clean' or 'conflict'")
    return PROCESSED_DIR / f"{mode}_chunks.jsonl"


def _load_chunks(mode: str) -> list[dict]:
    path = _chunks_path(mode)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


@lru_cache(maxsize=2)
def _load_bm25_index(mode: str):
    chunks = _load_chunks(mode)
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return chunks, BM25Okapi(tokenized_corpus)


def search_bm25(query: str, mode: str, top_n: int = BM25_TOP_N) -> list[dict]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    chunks, bm25 = _load_bm25_index(mode)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_n]

    results = []
    for index in ranked_indexes:
        chunk = chunks[index]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "score": float(scores[index]),
                "source_path": chunk["source_path"],
                "is_conflict_source": bool(chunk.get("is_conflict_source", False)),
            }
        )

    return results
