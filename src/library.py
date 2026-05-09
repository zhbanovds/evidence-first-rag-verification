from __future__ import annotations

import time
from typing import Any

from src.config import CHROMA_CLEAN_DIR, CHROMA_CONFLICT_DIR, TOP_K
from src.indexing.chroma_store import load_chroma_collection
from src.pipelines.final_evidence_first_rag import run_final_evidence_first_rag
from src.schemas import RAGRequest, RAGResponse
from src.verification.claim_extractor import extract_claims
from src.verification.claim_verifier import verify_claims


FINAL_PIPELINE_NAME = "final_evidence_first"

ALLOWED_MODES = ("clean", "conflict")


class VerifiedRAGSystem:
    """Public library facade for the final verified RAG system.

    The facade wraps only the Final Evidence-First Verified RAG pipeline. A
    user provides a question through :meth:`ask`, and the method returns a
    ``RAGResponse`` with the answer, retrieved chunks, fact plan, extracted
    claims, verification report, latency, call counters, and readable errors.

    Baseline, advanced retrieval, and post-hoc verification pipelines remain in
    ``src.pipelines`` for chapter 3 experiments, but they are not part of this
    public library API.
    """

    def __init__(self, mode: str = "clean"):
        if mode not in ALLOWED_MODES:
            allowed = ", ".join(ALLOWED_MODES)
            raise ValueError(f"Unknown mode '{mode}'. Allowed modes: {allowed}.")
        self.mode = mode

    def ask(self, request: RAGRequest | str) -> RAGResponse:
        """Run Final Evidence-First Verified RAG for a user question.

        Args:
            request: Either a plain question string or ``RAGRequest``. The
                deprecated ``RAGRequest.pipeline`` field must be omitted or set
                to ``"final_evidence_first"``.

        Returns:
            ``RAGResponse`` with the generated answer and structured internal
            artifacts. Runtime problems such as missing Chroma index or
            unavailable Ollama are reported in ``response.errors`` when they can
            be handled locally.
        """

        if isinstance(request, str):
            return self.run_pipeline(
                question=request,
                pipeline=FINAL_PIPELINE_NAME,
                mode=self.mode,
            )

        if not isinstance(request, RAGRequest):
            return self._error_response(
                question="",
                mode=self.mode,
                pipeline=FINAL_PIPELINE_NAME,
                error="request must be a RAGRequest instance or a question string.",
            )

        if request.pipeline not in {None, "", FINAL_PIPELINE_NAME}:
            return self._error_response(
                question=request.question,
                mode=request.mode,
                pipeline=str(request.pipeline),
                error=(
                    "RAGRequest.pipeline is deprecated in the library API. "
                    "VerifiedRAGSystem.ask() supports only "
                    f"'{FINAL_PIPELINE_NAME}'. Experimental pipelines are "
                    "available through scripts/04_run_main_eval.py and "
                    "scripts/05_run_conflict_eval.py."
                ),
            )

        return self.run_pipeline(
            question=request.question,
            pipeline=FINAL_PIPELINE_NAME,
            mode=request.mode,
            top_k=request.top_k,
        )

    def run_pipeline(
        self,
        question: str,
        pipeline: str = FINAL_PIPELINE_NAME,
        mode: str = "clean",
        top_k: int | None = None,
    ) -> RAGResponse:
        if pipeline != FINAL_PIPELINE_NAME:
            return self._error_response(
                question=question,
                mode=mode,
                pipeline=pipeline,
                error=(
                    "The library API is intended for the final verified "
                    f"pipeline only: '{FINAL_PIPELINE_NAME}'. Baseline, "
                    "advanced, and posthoc pipelines remain available through "
                    "the evaluation scripts."
                ),
            )

        if mode not in ALLOWED_MODES:
            allowed = ", ".join(ALLOWED_MODES)
            return self._error_response(
                question=question,
                mode=mode,
                pipeline=pipeline,
                error=f"Unknown mode '{mode}'. Allowed modes: {allowed}.",
            )

        if not str(question or "").strip():
            return self._error_response(
                question=question,
                mode=mode,
                pipeline=pipeline,
                error="Question must not be empty.",
            )

        if top_k is not None and top_k <= 0:
            return self._error_response(
                question=question,
                mode=mode,
                pipeline=pipeline,
                error="top_k must be a positive integer.",
            )

        index_error = self._check_chroma_index(mode)
        if index_error:
            return self._error_response(
                question=question,
                mode=mode,
                pipeline=pipeline,
                error=index_error,
            )

        started_at = time.perf_counter()

        try:
            kwargs = {
                "question_id": "LIB001",
                "question": question,
                "mode": mode,
            }
            if top_k is not None:
                kwargs["top_k"] = top_k
            else:
                kwargs["top_k"] = TOP_K

            result = run_final_evidence_first_rag(**kwargs)
        except Exception as exc:
            return self._error_response(
                question=question,
                mode=mode,
                pipeline=pipeline,
                error=f"Pipeline execution failed: {exc}",
                latency_sec=time.perf_counter() - started_at,
            )

        return self._response_from_result(result, question, mode, pipeline)

    def build_index(self, mode: str = "clean") -> None:
        """Explain how to build the Chroma index for library usage.

        Index creation is intentionally kept in ``scripts/02_build_index.py`` so
        that the library facade does not duplicate the experimental workflow.
        Run ``python3 scripts/02_build_index.py --mode clean`` or the conflict
        variant before calling :meth:`ask`.
        """

        if mode not in ALLOWED_MODES:
            allowed = ", ".join(ALLOWED_MODES)
            raise ValueError(f"Unknown mode '{mode}'. Allowed modes: {allowed}.")
        raise NotImplementedError(
            "Index building is intentionally kept in scripts/02_build_index.py. "
            f"Run: python3 scripts/02_build_index.py --mode {mode}"
        )

    def verify_answer(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
    ) -> list[dict]:
        """Verify an existing answer against supplied evidence chunks.

        This helper exposes the same claim extraction and LLM-as-a-judge
        verification modules used by the final pipeline. The input is the user
        question, an answer text, and evidence chunks; the output is a list of
        claim-level verdict dictionaries.
        """

        del question
        if not str(answer or "").strip():
            return [
                {
                    "claim": "",
                    "verdict": "NO_EVIDENCE",
                    "evidence_chunk_ids": [],
                    "explanation": "Answer must not be empty.",
                }
            ]

        if not evidence:
            return [
                {
                    "claim": "",
                    "verdict": "NO_EVIDENCE",
                    "evidence_chunk_ids": [],
                    "explanation": "Evidence chunks are required for verification.",
                }
            ]

        extraction_result = extract_claims(answer)
        if extraction_result.get("error"):
            return [
                {
                    "claim": "",
                    "verdict": "NO_EVIDENCE",
                    "evidence_chunk_ids": [],
                    "explanation": (
                        "Claim extraction failed: "
                        f"{extraction_result.get('error')}"
                    ),
                }
            ]

        claims = extraction_result.get("claims", [])
        if not claims:
            return []

        verification_result = verify_claims(claims, evidence)
        report = verification_result.get("verification_report", [])
        if report:
            return report

        if verification_result.get("error"):
            return [
                {
                    "claim": "",
                    "verdict": "NO_EVIDENCE",
                    "evidence_chunk_ids": [],
                    "explanation": (
                        "Claim verification failed: "
                        f"{verification_result.get('error')}"
                    ),
                }
            ]

        return []

    @staticmethod
    def _check_chroma_index(mode: str) -> str | None:
        chroma_dir = CHROMA_CLEAN_DIR if mode == "clean" else CHROMA_CONFLICT_DIR
        sqlite_path = chroma_dir / "chroma.sqlite3"
        if not sqlite_path.exists():
            return (
                f"Chroma index for mode '{mode}' was not found at {chroma_dir}. "
                f"Build it with: python3 scripts/02_build_index.py --mode {mode}"
            )

        try:
            collection = load_chroma_collection(mode)
            count = collection.count()
        except Exception as exc:
            return (
                f"Chroma index for mode '{mode}' could not be loaded: {exc}. "
                f"Rebuild it with: python3 scripts/02_build_index.py --mode {mode}"
            )

        if count <= 0:
            return (
                f"Chroma index for mode '{mode}' is empty. "
                f"Rebuild it with: python3 scripts/02_build_index.py --mode {mode}"
            )

        return None

    @staticmethod
    def _response_from_result(
        result: dict[str, Any],
        question: str,
        mode: str,
        pipeline: str,
    ) -> RAGResponse:
        final_answer = result.get("final_answer")
        if final_answer == "":
            final_answer = None
        elif final_answer is not None:
            final_answer = str(final_answer)

        return RAGResponse(
            question=str(result.get("question") or question),
            mode=str(result.get("mode") or mode),
            pipeline=str(result.get("pipeline") or pipeline),
            answer=str(result.get("answer") or ""),
            final_answer=final_answer,
            retrieved_chunks=_as_dict_list(result.get("retrieved_chunks")),
            fact_plan=_as_dict_list(result.get("fact_plan")),
            claims=_as_dict_list(result.get("claims")),
            verification_report=_as_dict_list(result.get("verification_report")),
            latency_sec=_as_float(result.get("latency_sec")),
            num_llm_calls=_as_int(result.get("num_llm_calls")),
            num_retrieval_calls=_as_int(result.get("num_retrieval_calls")),
            errors=_as_error_list(result.get("errors")),
        )

    @staticmethod
    def _error_response(
        question: str,
        mode: str,
        pipeline: str,
        error: str,
        latency_sec: float = 0.0,
    ) -> RAGResponse:
        return RAGResponse(
            question=question,
            mode=mode,
            pipeline=pipeline,
            latency_sec=latency_sec,
            errors=[error],
        )


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_error_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
