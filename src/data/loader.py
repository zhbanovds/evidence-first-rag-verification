from src.config import PROJECT_ROOT, RAW_CONFLICT_DIR, RAW_DIR


def _relative_path(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def _read_markdown_documents(paths, mode, is_conflict_source):
    documents = []
    for path in sorted(paths):
        documents.append(
            {
                "doc_id": path.stem,
                "source_path": _relative_path(path),
                "text": path.read_text(encoding="utf-8"),
                "mode": mode,
                "is_conflict_source": is_conflict_source,
            }
        )
    return documents


def load_documents(mode: str) -> list[dict]:
    if mode not in {"clean", "conflict"}:
        raise ValueError("mode must be 'clean' or 'conflict'")

    documents = _read_markdown_documents(
        RAW_DIR.glob("*.md"),
        mode=mode,
        is_conflict_source=False,
    )

    if mode == "conflict":
        documents.extend(
            _read_markdown_documents(
                RAW_CONFLICT_DIR.glob("*.md"),
                mode=mode,
                is_conflict_source=True,
            )
        )

    return documents
