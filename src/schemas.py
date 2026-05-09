from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGRequest:
    """Input for the public library API.

    The library facade always runs the Final Evidence-First Verified RAG
    pipeline. The pipeline field is kept only for backward compatibility with
    early examples and should not be used by new code.
    """

    question: str
    mode: str = "clean"
    top_k: int | None = None
    pipeline: str = "final_evidence_first"


@dataclass
class SourceChunk:
    chunk_id: str
    doc_id: str
    source_path: str | None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactPlanItem:
    claim_id: str
    claim: str
    evidence_chunk_ids: list[str]
    status: str


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    evidence_chunk_ids: list[str]
    explanation: str | None = None


@dataclass
class RAGResponse:
    question: str
    mode: str
    pipeline: str
    answer: str = ""
    final_answer: str | None = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    fact_plan: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    verification_report: list[dict[str, Any]] = field(default_factory=list)
    latency_sec: float = 0.0
    num_llm_calls: int = 0
    num_retrieval_calls: int = 0
    errors: list[str] = field(default_factory=list)
