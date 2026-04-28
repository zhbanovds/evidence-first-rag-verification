import time

from src.config import OLLAMA_TIMEOUT_SEC, TOP_K
from src.llm.ollama_client import generate_text
from src.prompts import ANSWER_PROMPT
from src.retrieval.hybrid_retriever import search_hybrid


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


def run_advanced_rag(
    question_id: str,
    question: str,
    mode: str = "clean",
    top_k: int = TOP_K,
) -> dict:
    started_at = time.perf_counter()
    result = {
        "question_id": question_id,
        "question": question,
        "pipeline": "advanced",
        "mode": mode,
        "retrieval_type": "hybrid",
        "retrieved_chunks": [],
        "answer": "",
        "final_answer": "",
        "claims": [],
        "verification_report": [],
        "latency_sec": 0.0,
        "num_llm_calls": 0,
        "num_retrieval_calls": 0,
        "errors": [],
    }

    try:
        retrieved_chunks = search_hybrid(question, mode=mode, top_k=top_k)
        result["retrieved_chunks"] = retrieved_chunks
        result["num_retrieval_calls"] = 2
    except Exception as exc:
        result["errors"].append(f"Retrieval failed: {exc}")
        result["latency_sec"] = time.perf_counter() - started_at
        return result

    context = _format_context(result["retrieved_chunks"])
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    llm_result = generate_text(prompt, timeout_sec=OLLAMA_TIMEOUT_SEC, num_predict=256)
    result["num_llm_calls"] = 1

    if llm_result["error"]:
        result["errors"].append(llm_result["error"])
    else:
        answer = llm_result["text"].strip()
        result["answer"] = answer
        result["final_answer"] = answer

    result["latency_sec"] = time.perf_counter() - started_at
    return result
