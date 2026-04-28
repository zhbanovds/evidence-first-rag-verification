import json
import re
from typing import Any, Optional


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_with_decoder(text: str) -> str:
    decoder = json.JSONDecoder()
    stripped = text.strip()

    for start, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        return stripped[start : start + end]

    return ""


def extract_json_from_text(text: str) -> str:
    if not text:
        return ""

    stripped = text.strip()
    fenced_blocks = FENCED_JSON_RE.findall(stripped)
    for block in fenced_blocks:
        extracted = _extract_with_decoder(block)
        if extracted:
            return extracted

    return _extract_with_decoder(stripped)


def safe_json_loads(text: str, fallback: Any) -> tuple[Any, Optional[str]]:
    json_text = extract_json_from_text(text)
    if not json_text:
        return fallback, "JSON object or array not found in LLM output."

    try:
        return json.loads(json_text), None
    except json.JSONDecodeError as exc:
        return fallback, f"Invalid JSON: {exc}"
