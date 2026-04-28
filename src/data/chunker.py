from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        previous_start = start
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(text):
            break

        start = max(end - overlap, 0)
        if start <= previous_start:
            start = end
        while start < len(text) and text[start].isspace():
            start += 1

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []

    for document in documents:
        text_chunks = _split_text(
            document["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for index, text in enumerate(text_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}__{index:04d}",
                    "doc_id": document["doc_id"],
                    "source_path": document["source_path"],
                    "text": text,
                    "chunk_index": index,
                    "mode": document["mode"],
                    "is_conflict_source": document["is_conflict_source"],
                }
            )

    return chunks
