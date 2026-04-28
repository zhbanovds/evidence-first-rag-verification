import time

from src.config import OLLAMA_TIMEOUT_SEC, TOP_K
from src.llm.ollama_client import generate_text
from src.prompts import ANSWER_PROMPT
from src.retrieval.hybrid_retriever import search_hybrid
from src.verification.claim_extractor import extract_claims
from src.verification.claim_verifier import verify_claims


def _format_context(chunks: list[dict]) -> str:
    formatted_chunks = []
    for rank, chunk in enumerate(chunks, start=1):
        formatted_chunks.append(
            "\n".join(
                [
                    f"[{rank}] chunk_id: {chunk.get('chunk_id')}",
                    f"doc_id: {chunk.get('doc_id')}",
                    "text:",
                    chunk.get("text", ""),
                ]
            )
        )
    return "\n\n---\n\n".join(formatted_chunks)


def run_posthoc_verification_rag(
    question_id: str,
    question: str,
    mode: str = "clean",
    top_k: int = TOP_K,
) -> dict:
    result = {
        "question_id": question_id,
        "question": question,
        "pipeline": "posthoc",
        "mode": mode,
        "retrieval_type": "hybrid",
        "retrieved_chunks": [],
        "fact_plan": [],
        "answer": "",
        "claims": [],
        "verification_report": [],
        "final_answer": "",
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

    context = _format_context(result["retrieved_chunks"])
    answer_prompt = ANSWER_PROMPT.format(context=context, question=question)
    answer_result = generate_text(
        answer_prompt,
        timeout_sec=OLLAMA_TIMEOUT_SEC,
        num_predict=256,
    )
    result["num_llm_calls"] += 1
    result["latency_sec"] += answer_result.get("latency_sec", 0.0)

    if answer_result.get("error"):
        result["errors"].append(f"Answer generation failed: {answer_result['error']}")
        result["final_answer"] = result["answer"]
        return result

    answer = answer_result.get("text", "").strip()
    result["answer"] = answer
    result["final_answer"] = answer

    extraction_result = extract_claims(answer)
    result["num_llm_calls"] += extraction_result.get("num_llm_calls", 0)
    result["latency_sec"] += extraction_result.get("latency_sec", 0.0)
    result["claims"] = extraction_result.get("claims", [])

    if extraction_result.get("error"):
        result["errors"].append(f"Claim extraction failed: {extraction_result['error']}")

    verification_result = verify_claims(result["claims"], result["retrieved_chunks"])
    result["num_llm_calls"] += verification_result.get("num_llm_calls", 0)
    result["latency_sec"] += verification_result.get("latency_sec", 0.0)
    result["verification_report"] = verification_result.get("verification_report", [])

    if verification_result.get("error"):
        result["errors"].append(f"Claim verification failed: {verification_result['error']}")

    result["final_answer"] = result["answer"]
    return result
