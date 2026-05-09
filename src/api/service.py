from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from src.config import CHROMA_CLEAN_DIR, CHROMA_CONFLICT_DIR, OLLAMA_URL
from src.library import VerifiedRAGSystem
from src.schemas import RAGRequest, RAGResponse


SERVICE_NAME = "rag-vkr-api"
SERVICE_VERSION = "0.1.0"

logger = logging.getLogger(__name__)


class RAGAPIService:
    """Service layer for the demonstration HTTP wrapper."""

    def __init__(self) -> None:
        self._systems: dict[str, VerifiedRAGSystem] = {}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "library_available": True,
            "chroma_clean_index_exists": self._index_exists("clean"),
            "chroma_conflict_index_exists": self._index_exists("conflict"),
            "ollama_configured": bool(str(OLLAMA_URL).strip()),
        }

    def ask(self, question: str, mode: str = "clean", top_k: int | None = None) -> dict:
        try:
            rag = self._get_system(mode)
            response = rag.ask(
                RAGRequest(
                    question=question,
                    mode=mode,
                    top_k=top_k,
                )
            )
            return self._response_to_dict(response)
        except Exception as exc:
            logger.exception("RAG ask request failed")
            return self._error_result(
                question=question,
                mode=mode,
                error=f"API request failed: {exc}",
            )

    def verify(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
        mode: str = "clean",
    ) -> dict:
        try:
            rag = self._get_system(mode)
            report = rag.verify_answer(question, answer, evidence)
            return {
                "question": question,
                "verification_report": report,
                "errors": [],
            }
        except NotImplementedError as exc:
            return {
                "question": question,
                "verification_report": [],
                "errors": [str(exc)],
            }
        except Exception as exc:
            logger.exception("Answer verification failed")
            return {
                "question": question,
                "verification_report": [],
                "errors": [f"Answer verification failed: {exc}"],
            }

    def index_message(self, mode: str) -> dict[str, str]:
        return {
            "error": "Index building is not exposed by this demonstration API.",
            "detail": (
                "Build the index from the project root with: "
                f"python3 scripts/02_build_index.py --mode {mode}"
            ),
        }

    def _get_system(self, mode: str) -> VerifiedRAGSystem:
        if mode not in self._systems:
            self._systems[mode] = VerifiedRAGSystem(mode=mode)
        return self._systems[mode]

    @staticmethod
    def _index_exists(mode: str) -> bool:
        chroma_dir = CHROMA_CLEAN_DIR if mode == "clean" else CHROMA_CONFLICT_DIR
        return (chroma_dir / "chroma.sqlite3").exists()

    @staticmethod
    def _response_to_dict(response: RAGResponse) -> dict[str, Any]:
        if is_dataclass(response):
            return asdict(response)
        return dict(response)

    @staticmethod
    def _error_result(question: str, mode: str, error: str) -> dict[str, Any]:
        return {
            "question": question,
            "mode": mode,
            "pipeline": "final_evidence_first",
            "answer": "",
            "final_answer": None,
            "retrieved_chunks": [],
            "fact_plan": [],
            "claims": [],
            "verification_report": [],
            "latency_sec": 0.0,
            "num_llm_calls": 0,
            "num_retrieval_calls": 0,
            "errors": [error],
        }


api_service = RAGAPIService()
