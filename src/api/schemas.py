from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, validator


ALLOWED_MODES = {"clean", "conflict"}


class AskRequest(BaseModel):
    question: str = Field(..., description="User question for the RAG system.")
    mode: str = Field("clean", description="Knowledge base mode: clean or conflict.")
    top_k: Optional[int] = Field(
        None,
        description="Optional retrieval top-k override.",
    )

    @validator("question")
    def ask_question_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @validator("mode")
    def ask_mode_must_be_supported(cls, value: str) -> str:
        if value not in ALLOWED_MODES:
            raise ValueError("mode must be 'clean' or 'conflict'")
        return value

    @validator("top_k")
    def ask_top_k_must_be_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("top_k must be positive")
        return value


class VerifyRequest(BaseModel):
    question: str
    answer: str
    evidence: list[dict[str, Any]]

    @validator("question")
    def verify_question_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @validator("answer")
    def verify_answer_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be empty")
        return value


class IndexRequest(BaseModel):
    mode: str = "clean"

    @validator("mode")
    def index_mode_must_be_supported(cls, value: str) -> str:
        if value not in ALLOWED_MODES:
            raise ValueError("mode must be 'clean' or 'conflict'")
        return value


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    library_available: bool
    chroma_clean_index_exists: bool
    chroma_conflict_index_exists: bool
    ollama_configured: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
