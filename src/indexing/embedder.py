from functools import lru_cache
import os

from src.config import EMBEDDING_CACHE_DIR, EMBEDDING_MODEL


ERROR_MESSAGE = (
    "Embedding model is not cached. Run with --allow-download once to download it."
)


def _local_model_dir():
    return EMBEDDING_CACHE_DIR / EMBEDDING_MODEL.replace("/", "__")


def _set_offline_mode(enabled: bool) -> None:
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def _load_sentence_transformer(*args, **kwargs):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(*args, **kwargs)


@lru_cache(maxsize=1)
def get_embedding_model(local_files_only: bool = True, show_progress: bool = False):
    _set_offline_mode(local_files_only)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    local_model_dir = _local_model_dir()

    if local_files_only:
        if local_model_dir.exists():
            try:
                return _load_sentence_transformer(
                    str(local_model_dir),
                    local_files_only=True,
                )
            except Exception:
                pass

        try:
            return _load_sentence_transformer(
                EMBEDDING_MODEL,
                cache_folder=str(EMBEDDING_CACHE_DIR),
                local_files_only=True,
            )
        except Exception:
            pass

        try:
            return _load_sentence_transformer(
                EMBEDDING_MODEL,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(ERROR_MESSAGE) from exc

    if show_progress:
        print(f"[embedder] Downloading embedding model: {EMBEDDING_MODEL}")

    model = _load_sentence_transformer(
        EMBEDDING_MODEL,
        cache_folder=str(EMBEDDING_CACHE_DIR),
        local_files_only=False,
    )
    model.save(str(local_model_dir))
    return model


def embed_texts(
    texts: list[str],
    local_files_only: bool = True,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    model = get_embedding_model(
        local_files_only=local_files_only,
        show_progress=show_progress_bar,
    )
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return embeddings.astype(float).tolist()
