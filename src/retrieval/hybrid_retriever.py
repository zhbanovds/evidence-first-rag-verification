from src.config import BM25_TOP_N, RRF_K, TOP_K, VECTOR_TOP_N
from src.retrieval.bm25_retriever import search_bm25
from src.retrieval.vector_retriever import search_vector


def _merge_result(
    merged: dict,
    chunk: dict,
    source: str,
    rank: int,
    rrf_k: int,
) -> None:
    chunk_id = chunk["chunk_id"]
    if chunk_id not in merged:
        merged[chunk_id] = {
            **chunk,
            "score": 0.0,
            "rrf_score": 0.0,
            "vector_score": None,
            "bm25_score": None,
            "vector_rank": None,
            "bm25_rank": None,
        }

    contribution = 1.0 / (rrf_k + rank)
    merged[chunk_id]["rrf_score"] += contribution
    merged[chunk_id]["score"] = merged[chunk_id]["rrf_score"]

    if source == "vector":
        merged[chunk_id]["vector_score"] = chunk.get("score")
        merged[chunk_id]["vector_rank"] = rank
    elif source == "bm25":
        merged[chunk_id]["bm25_score"] = chunk.get("score")
        merged[chunk_id]["bm25_rank"] = rank
    else:
        raise ValueError("source must be 'vector' or 'bm25'")


def search_hybrid(
    query: str,
    mode: str,
    top_k: int = TOP_K,
    vector_top_n: int = VECTOR_TOP_N,
    bm25_top_n: int = BM25_TOP_N,
    rrf_k: int = RRF_K,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if vector_top_n <= 0:
        raise ValueError("vector_top_n must be positive")
    if bm25_top_n <= 0:
        raise ValueError("bm25_top_n must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    vector_results = search_vector(query, mode=mode, top_k=vector_top_n)
    bm25_results = search_bm25(query, mode=mode, top_n=bm25_top_n)

    merged = {}
    for rank, chunk in enumerate(vector_results, start=1):
        _merge_result(merged, chunk, source="vector", rank=rank, rrf_k=rrf_k)
    for rank, chunk in enumerate(bm25_results, start=1):
        _merge_result(merged, chunk, source="bm25", rank=rank, rrf_k=rrf_k)

    return sorted(
        merged.values(),
        key=lambda chunk: chunk["rrf_score"],
        reverse=True,
    )[:top_k]
