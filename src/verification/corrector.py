import json

from src.config import OLLAMA_TIMEOUT_SEC
from src.llm.ollama_client import generate_text
from src.prompts import CORRECTION_PROMPT


def _call_llm(prompt: str, llm_client=None) -> dict:
    if llm_client is None:
        return generate_text(prompt, timeout_sec=OLLAMA_TIMEOUT_SEC, num_predict=384)

    if callable(llm_client):
        response = llm_client(prompt)
    elif hasattr(llm_client, "generate_text"):
        response = llm_client.generate_text(prompt)
    else:
        raise TypeError("llm_client must be callable or have generate_text(prompt).")

    if isinstance(response, dict):
        return response

    return {
        "text": str(response),
        "latency_sec": 0.0,
        "error": None,
    }


def correct_answer(draft_answer, verification_report, llm_client=None) -> dict:
    result = {
        "final_answer": "",
        "raw_output": "",
        "error": None,
        "latency_sec": 0.0,
        "num_llm_calls": 1,
    }

    prompt = CORRECTION_PROMPT.format(
        question="Не указан. Исправь ответ только по draft answer и verification report.",
        draft_answer=draft_answer,
        verification_report=json.dumps(
            verification_report,
            ensure_ascii=False,
            indent=2,
        ),
    )

    try:
        llm_result = _call_llm(prompt, llm_client=llm_client)
    except Exception as exc:
        result["error"] = f"Correction LLM call failed: {exc}"
        return result

    raw_output = llm_result.get("text", "")
    result["raw_output"] = raw_output
    result["latency_sec"] = llm_result.get("latency_sec", 0.0)

    if llm_result.get("error"):
        result["error"] = llm_result["error"]
        return result

    result["final_answer"] = raw_output.strip()
    return result
