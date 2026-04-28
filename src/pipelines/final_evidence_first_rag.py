import time

from src.config import TOP_K
from src.planning.answer_from_plan import (
    check_fact_plan_usage,
    generate_answer_from_fact_plan,
)
from src.planning.evidence_planner import build_fact_plan
from src.retrieval.hybrid_retriever import search_hybrid
from src.verification.claim_extractor import extract_claims
from src.verification.claim_verifier import verify_claims
from src.verification.corrector import correct_answer


def _build_verification_evidence(
    fact_plan: list[dict],
    prepared_chunks: list[dict],
    retrieved_chunks: list[dict],
) -> list[dict]:
    chunks_by_id = {
        str(chunk.get("chunk_id")): chunk
        for chunk in [*prepared_chunks, *retrieved_chunks]
        if chunk.get("chunk_id")
    }
    ordered = []
    seen = set()

    def add_chunk(chunk_id: str) -> None:
        if not chunk_id or chunk_id in seen:
            return
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            return
        seen.add(chunk_id)
        ordered.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text", ""),
            }
        )

    for item in fact_plan:
        evidence_ids = item.get("evidence_chunk_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        for chunk_id in evidence_ids:
            add_chunk(str(chunk_id))

    if ordered:
        return ordered

    for chunk in prepared_chunks:
        add_chunk(str(chunk.get("chunk_id", "")))

    for chunk in retrieved_chunks:
        add_chunk(str(chunk.get("chunk_id", "")))

    return ordered


def _claim_like_sentence_count(text: str) -> int:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return 0
    parts = [
        part.strip()
        for part in normalized.replace("\n", " ").split(".")
        if part.strip()
    ]
    return len(parts) if parts else 1


def _supported_claim_count(verification_report: list[dict]) -> int:
    supported_verdicts = {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONFLICT"}
    return sum(
        1
        for item in verification_report
        if str(item.get("verdict", "")).upper() in supported_verdicts
    )


def _apply_answer_usage_report(result: dict, usage_report: dict, prefix: str) -> None:
    result[f"fact_plan_used_in_answer_rate_{prefix}"] = usage_report.get(
        "fact_plan_used_in_answer_rate",
        0.0,
    )
    if prefix == "after_retry":
        result["answer_coverage_gate_passed"] = bool(
            usage_report.get("answer_coverage_gate_passed", False)
        )
        result["answer_coverage_gate_reason"] = usage_report.get(
            "answer_coverage_gate_reason",
            "",
        )
        result["missing_fact_plan_claim_ids_in_answer"] = usage_report.get(
            "missing_claim_ids",
            [],
        )


def run_final_evidence_first_rag(
    question_id: str,
    question: str,
    mode: str = "clean",
    top_k: int = TOP_K,
) -> dict:
    result = {
        "question_id": question_id,
        "question": question,
        "pipeline": "final_evidence_first",
        "mode": mode,
        "retrieval_type": "hybrid",
        "retrieved_chunks": [],
        "fact_plan": [],
        "answer": "",
        "claims": [],
        "verification_report": [],
        "final_answer": "",
        "answer_from_plan_fallback_used": False,
        "question_intent": "default",
        "question_focus_terms": [],
        "comparison_sides": [],
        "numeric_targets": [],
        "expected_entity_type": "general",
        "fact_plan_retry_used": False,
        "coverage_gate_passed": False,
        "coverage_gate_reason": "",
        "fact_plan_used_in_answer_rate_before_retry": 0.0,
        "fact_plan_used_in_answer_rate_after_retry": 0.0,
        "fact_plan_used_in_final_answer_rate": 0.0,
        "answer_coverage_gate_passed": False,
        "answer_coverage_gate_reason": "",
        "answer_from_plan_retry_used": False,
        "missing_fact_plan_claim_ids_in_answer": [],
        "missing_fact_plan_claim_ids_in_final_answer": [],
        "final_supported_claim_count": 0,
        "final_answer_claim_count": 0,
        "final_answer_from_plan_missing_count": 0,
        "prepared_evidence_count": 0,
        "prepared_evidence_chunk_ids": [],
        "neighbor_expansion_used": False,
        "latency_sec": 0.0,
        "num_llm_calls": 0,
        "num_retrieval_calls": 0,
        "errors": [],
    }

    retrieval_started_at = time.perf_counter()
    try:
        retrieved_chunks = search_hybrid(question, mode=mode, top_k=top_k)
        result["retrieved_chunks"] = retrieved_chunks
        result["num_retrieval_calls"] = 2
    except Exception as exc:
        result["errors"].append(f"Retrieval failed: {exc}")
        result["latency_sec"] += time.perf_counter() - retrieval_started_at
        return result
    result["latency_sec"] += time.perf_counter() - retrieval_started_at

    fact_plan_result = build_fact_plan(
        question,
        result["retrieved_chunks"],
        mode=mode,
    )
    result["num_llm_calls"] += fact_plan_result.get("num_llm_calls", 0)
    result["latency_sec"] += fact_plan_result.get("latency_sec", 0.0)
    result["fact_plan"] = fact_plan_result.get("fact_plan", [])
    prepared_evidence_chunks = fact_plan_result.get("prepared_evidence_chunks", [])
    result["question_intent"] = fact_plan_result.get("question_intent", "default")
    result["question_focus_terms"] = fact_plan_result.get("question_focus_terms", [])
    result["comparison_sides"] = fact_plan_result.get("comparison_sides", [])
    result["numeric_targets"] = fact_plan_result.get("numeric_targets", [])
    result["expected_entity_type"] = fact_plan_result.get(
        "expected_entity_type",
        "general",
    )
    result["fact_plan_retry_used"] = bool(
        fact_plan_result.get("fact_plan_retry_used", False)
    )
    result["coverage_gate_passed"] = bool(
        fact_plan_result.get("coverage_gate_passed", False)
    )
    result["coverage_gate_reason"] = fact_plan_result.get(
        "coverage_gate_reason",
        "",
    )
    result["prepared_evidence_count"] = fact_plan_result.get(
        "prepared_evidence_count",
        len(fact_plan_result.get("prepared_evidence_chunks", [])),
    )
    result["prepared_evidence_chunk_ids"] = fact_plan_result.get(
        "prepared_evidence_chunk_ids",
        [
            str(chunk.get("chunk_id"))
            for chunk in fact_plan_result.get("prepared_evidence_chunks", [])
            if chunk.get("chunk_id")
        ],
    )
    result["neighbor_expansion_used"] = bool(
        fact_plan_result.get("neighbor_expansion_used", False)
    )

    if fact_plan_result.get("error"):
        result["errors"].append(f"Fact planning failed: {fact_plan_result['error']}")

    answer_result = generate_answer_from_fact_plan(question, result["fact_plan"])
    result["num_llm_calls"] += answer_result.get("num_llm_calls", 0)
    result["latency_sec"] += answer_result.get("latency_sec", 0.0)
    result["answer"] = answer_result.get("answer", "")
    result["final_answer"] = result["answer"]
    result["answer_from_plan_fallback_used"] = bool(
        answer_result.get("fallback_used", False)
    )

    if answer_result.get("error"):
        result["errors"].append(
            f"Answer from fact plan failed: {answer_result['error']}"
        )
        return result

    initial_usage_report = check_fact_plan_usage(
        question,
        result["fact_plan"],
        result["answer"],
    )
    _apply_answer_usage_report(result, initial_usage_report, "before_retry")

    final_draft_usage_report = initial_usage_report
    if not initial_usage_report.get("answer_coverage_gate_passed", False):
        retry_result = generate_answer_from_fact_plan(
            question,
            result["fact_plan"],
            strict=True,
            coverage_feedback=initial_usage_report,
        )
        result["num_llm_calls"] += retry_result.get("num_llm_calls", 0)
        result["latency_sec"] += retry_result.get("latency_sec", 0.0)
        result["answer_from_plan_retry_used"] = True
        result["answer_from_plan_fallback_used"] = bool(
            result["answer_from_plan_fallback_used"]
            or retry_result.get("fallback_used", False)
        )

        if retry_result.get("error"):
            result["errors"].append(
                f"Answer coverage retry failed: {retry_result['error']}"
            )
        else:
            result["answer"] = retry_result.get("answer", "") or result["answer"]
            result["final_answer"] = result["answer"]
            final_draft_usage_report = check_fact_plan_usage(
                question,
                result["fact_plan"],
                result["answer"],
            )

    _apply_answer_usage_report(result, final_draft_usage_report, "after_retry")
    result["final_answer_claim_count"] = _claim_like_sentence_count(
        result["final_answer"]
    )

    if not result["fact_plan"]:
        return result

    extraction_result = extract_claims(result["answer"])
    result["num_llm_calls"] += extraction_result.get("num_llm_calls", 0)
    result["latency_sec"] += extraction_result.get("latency_sec", 0.0)
    result["claims"] = extraction_result.get("claims", [])

    if extraction_result.get("error"):
        result["errors"].append(f"Claim extraction failed: {extraction_result['error']}")
        return result

    if not result["claims"]:
        return result

    verification_evidence = _build_verification_evidence(
        result["fact_plan"],
        prepared_evidence_chunks,
        result["retrieved_chunks"],
    )
    verification_result = verify_claims(result["claims"], verification_evidence)
    result["num_llm_calls"] += verification_result.get("num_llm_calls", 0)
    result["latency_sec"] += verification_result.get("latency_sec", 0.0)
    result["verification_report"] = verification_result.get("verification_report", [])
    result["final_supported_claim_count"] = _supported_claim_count(
        result["verification_report"]
    )

    if verification_result.get("error"):
        result["errors"].append(
            f"Claim verification failed: {verification_result['error']}"
        )

    correction_result = correct_answer(result["answer"], result["verification_report"])
    result["num_llm_calls"] += correction_result.get("num_llm_calls", 0)
    result["latency_sec"] += correction_result.get("latency_sec", 0.0)

    if correction_result.get("error"):
        result["errors"].append(f"Correction failed: {correction_result['error']}")
        result["final_answer"] = result["answer"]
    else:
        result["final_answer"] = correction_result.get("final_answer") or result["answer"]

    final_usage_report = check_fact_plan_usage(
        question,
        result["fact_plan"],
        result["final_answer"],
    )
    result["fact_plan_used_in_final_answer_rate"] = final_usage_report.get(
        "fact_plan_used_in_answer_rate",
        0.0,
    )
    result["missing_fact_plan_claim_ids_in_final_answer"] = final_usage_report.get(
        "missing_claim_ids",
        [],
    )
    result["final_answer_from_plan_missing_count"] = len(
        result["missing_fact_plan_claim_ids_in_final_answer"]
    )
    result["final_answer_claim_count"] = _claim_like_sentence_count(
        result["final_answer"]
    )

    return result
