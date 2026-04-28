import json
from typing import Optional

from src.config import MAX_CLAIMS_TO_VERIFY, OLLAMA_TIMEOUT_SEC
from src.llm.ollama_client import generate_text
from src.prompts import CLAIM_VERIFICATION_PROMPT
from src.utils import safe_json_loads


ALLOWED_VERDICTS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "NOT_SUPPORTED",
    "CONFLICT",
    "NO_EVIDENCE",
}


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


def _format_evidence(evidence_chunks: list[dict]) -> str:
    formatted_chunks = []
    for rank, chunk in enumerate(evidence_chunks, start=1):
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


def _empty_report_item(claim: dict, raw_output: str = "", error: Optional[str] = None) -> dict:
    return {
        "claim_id": str(claim.get("claim_id", "")),
        "claim": str(claim.get("claim", "")),
        "verdict": "NO_EVIDENCE",
        "evidence_chunk_id": None,
        "explanation": "",
        "raw_output": raw_output,
        "error": error,
    }


def _normalize_report_item(claim: dict, parsed: dict, raw_output: str) -> dict:
    error = None
    verdict = str(parsed.get("verdict", "NO_EVIDENCE")).strip().upper()
    if verdict not in ALLOWED_VERDICTS:
        error = f"Unknown verdict '{verdict}', replaced with NO_EVIDENCE."
        verdict = "NO_EVIDENCE"

    evidence_ids = parsed.get("evidence_chunk_ids", [])
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids]
    if not isinstance(evidence_ids, list):
        evidence_ids = []

    evidence_ids = [str(chunk_id) for chunk_id in evidence_ids if chunk_id]

    return {
        "claim_id": str(parsed.get("claim_id") or claim.get("claim_id", "")),
        "claim": str(parsed.get("claim") or claim.get("claim", "")),
        "verdict": verdict,
        "evidence_chunk_id": evidence_ids[0] if evidence_ids else None,
        "evidence_chunk_ids": evidence_ids,
        "explanation": str(parsed.get("explanation", "")),
        "raw_output": raw_output,
        "error": error,
    }


def verify_claims(claims, evidence_chunks, llm_client=None) -> dict:
    claims_to_verify = claims[:MAX_CLAIMS_TO_VERIFY]
    skipped_claims = claims[MAX_CLAIMS_TO_VERIFY:]
    result = {
        "verification_report": [],
        "raw_outputs": [],
        "error": None,
        "latency_sec": 0.0,
        "num_llm_calls": 0,
        "max_claims_to_verify": MAX_CLAIMS_TO_VERIFY,
        "skipped_claims_count": len(skipped_claims),
        "skipped_claims": skipped_claims,
    }
    evidence = _format_evidence(evidence_chunks)
    errors = []

    for claim in claims_to_verify:
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            claim_id=claim.get("claim_id", ""),
            claim=claim.get("claim", ""),
            evidence=evidence,
        )

        result["num_llm_calls"] += 1
        try:
            llm_result = _call_llm(prompt, llm_client=llm_client)
        except Exception as exc:
            error = f"Claim verification LLM call failed for {claim.get('claim_id')}: {exc}"
            errors.append(error)
            result["verification_report"].append(_empty_report_item(claim, error=error))
            continue

        raw_output = llm_result.get("text", "")
        latency_sec = llm_result.get("latency_sec", 0.0)
        result["latency_sec"] += latency_sec
        result["raw_outputs"].append(
            {
                "claim_id": claim.get("claim_id"),
                "raw_output": raw_output,
                "latency_sec": latency_sec,
                "error": llm_result.get("error"),
            }
        )

        if llm_result.get("error"):
            error = llm_result["error"]
            errors.append(f"{claim.get('claim_id')}: {error}")
            result["verification_report"].append(
                _empty_report_item(claim, raw_output=raw_output, error=error)
            )
            continue

        parsed, parse_error = safe_json_loads(raw_output, fallback={})
        if parse_error:
            errors.append(f"{claim.get('claim_id')}: {parse_error}")
            result["verification_report"].append(
                _empty_report_item(claim, raw_output=raw_output, error=parse_error)
            )
            continue

        if not isinstance(parsed, dict):
            error = "Parsed JSON is not an object."
            errors.append(f"{claim.get('claim_id')}: {error}")
            result["verification_report"].append(
                _empty_report_item(claim, raw_output=raw_output, error=error)
            )
            continue

        report_item = _normalize_report_item(claim, parsed, raw_output)
        if report_item["error"]:
            errors.append(f"{claim.get('claim_id')}: {report_item['error']}")
        result["verification_report"].append(report_item)

    if errors:
        result["error"] = "; ".join(errors)

    return result
