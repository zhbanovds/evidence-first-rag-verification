from src.config import TOP_K
from src.indexing.chroma_store import load_chroma_collection
from src.indexing.embedder import embed_texts


def _score_from_distance(distance):
    if distance is None:
        return None
    return 1.0 - float(distance)


def search_vector(query: str, mode: str, top_k: int = TOP_K) -> list[dict]:
    if mode not in {"clean", "conflict"}:
        raise ValueError("mode must be 'clean' or 'conflict'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    collection = load_chroma_collection(mode)
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        chunks.append(
            {
                "chunk_id": metadata.get("chunk_id", chunk_id),
                "doc_id": metadata.get("doc_id"),
                "text": text,
                "score": _score_from_distance(distance),
                "source_path": metadata.get("source_path"),
                "is_conflict_source": bool(metadata.get("is_conflict_source", False)),
            }
        )

    return chunks
