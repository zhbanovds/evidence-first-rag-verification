from chromadb import PersistentClient

from src.config import CHROMA_CLEAN_DIR, CHROMA_CONFLICT_DIR
from src.indexing.embedder import embed_texts


COLLECTION_NAMES = {
    "clean": "mai_rag_clean",
    "conflict": "mai_rag_conflict",
}


def _get_chroma_dir(mode: str):
    if mode == "clean":
        return CHROMA_CLEAN_DIR
    if mode == "conflict":
        return CHROMA_CONFLICT_DIR
    raise ValueError("mode must be 'clean' or 'conflict'")


def _validate_chunks(chunks: list[dict], mode: str) -> None:
    for chunk in chunks:
        source_path = chunk.get("source_path", "")
        if source_path.startswith("data/eval/"):
            raise ValueError(f"Eval file cannot be indexed: {source_path}")
        if chunk.get("mode") != mode:
            raise ValueError(
                f"Chunk {chunk.get('chunk_id')} has mode={chunk.get('mode')}, expected {mode}"
            )


def _metadata_from_chunk(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "source_path": chunk["source_path"],
        "mode": chunk["mode"],
        "is_conflict_source": bool(chunk["is_conflict_source"]),
        "chunk_index": int(chunk["chunk_index"]),
    }


def load_chroma_collection(mode: str):
    if mode not in COLLECTION_NAMES:
        raise ValueError("mode must be 'clean' or 'conflict'")

    client = PersistentClient(path=str(_get_chroma_dir(mode)))
    return client.get_collection(COLLECTION_NAMES[mode])


def build_chroma_index(
    chunks: list[dict],
    mode: str,
    local_files_only: bool = True,
):
    if mode not in COLLECTION_NAMES:
        raise ValueError("mode must be 'clean' or 'conflict'")

    _validate_chunks(chunks, mode)

    chroma_dir = _get_chroma_dir(mode)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(chroma_dir))
    collection_name = COLLECTION_NAMES[mode]

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(
            texts,
            local_files_only=local_files_only,
            show_progress_bar=True,
        )

        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_metadata_from_chunk(chunk) for chunk in batch],
        )

    return collection
