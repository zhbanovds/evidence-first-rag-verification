from typing import Optional

from src.config import OLLAMA_TIMEOUT_SEC
from src.llm.ollama_client import generate_text
from src.prompts import CLAIM_EXTRACTION_PROMPT
from src.utils import safe_json_loads


def _call_llm(prompt: str, llm_client=None) -> dict:
    if llm_client is None:
        return generate_text(prompt, timeout_sec=OLLAMA_TIMEOUT_SEC, num_predict=512)

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


def _normalize_claims(parsed) -> tuple[list[dict], Optional[str]]:
    if isinstance(parsed, dict):
        for key in ("claims", "claim_list", "result"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
        else:
            if "claim" in parsed:
                parsed = [parsed]

    if not isinstance(parsed, list):
        return [], "Parsed JSON is not a list."

    claims = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue

        claim_id = str(item.get("claim_id") or f"C{index:03d}").strip()
        claims.append(
            {
                "claim_id": claim_id,
                "claim": claim,
            }
        )

    return claims, None


def extract_claims(answer: str, llm_client=None) -> dict:
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    result = {
        "claims": [],
        "raw_output": "",
        "error": None,
        "latency_sec": 0.0,
        "num_llm_calls": 1,
    }

    try:
        llm_result = _call_llm(prompt, llm_client=llm_client)
    except Exception as exc:
        result["error"] = f"Claim extraction LLM call failed: {exc}"
        return result

    raw_output = llm_result.get("text", "")
    result["raw_output"] = raw_output
    result["latency_sec"] = llm_result.get("latency_sec", 0.0)

    if llm_result.get("error"):
        result["error"] = llm_result["error"]
        return result

    parsed, parse_error = safe_json_loads(raw_output, fallback=[])
    if parse_error:
        result["error"] = parse_error
        return result

    claims, normalize_error = _normalize_claims(parsed)
    result["claims"] = claims
    result["error"] = normalize_error
    return result
