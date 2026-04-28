import time
from typing import Optional

import requests

from src.config import LLM_MODEL, OLLAMA_TIMEOUT_SEC, OLLAMA_URL, TEMPERATURE


def generate_text(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    timeout_sec: int = OLLAMA_TIMEOUT_SEC,
    num_predict: Optional[int] = None,
    think: bool = False,
) -> dict:
    started_at = time.perf_counter()
    options = {
        "temperature": temperature,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": options,
    }

    result = {
        "text": "",
        "latency_sec": 0.0,
        "model": model,
        "error": None,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        result["error"] = f"Ollama request timed out: {exc}"
        result["latency_sec"] = time.perf_counter() - started_at
        return result
    except requests.exceptions.ConnectionError as exc:
        result["error"] = f"Ollama is unavailable: {exc}"
        result["latency_sec"] = time.perf_counter() - started_at
        return result
    except requests.exceptions.RequestException as exc:
        result["error"] = f"Ollama request failed: {exc}"
        result["latency_sec"] = time.perf_counter() - started_at
        return result

    try:
        data = response.json()
    except ValueError as exc:
        result["error"] = f"Ollama returned invalid JSON: {exc}"
        result["latency_sec"] = time.perf_counter() - started_at
        return result

    text = data.get("response", "")
    if not text:
        result["error"] = "Ollama returned an empty response."
        result["latency_sec"] = time.perf_counter() - started_at
        return result

    result["text"] = text
    result["latency_sec"] = time.perf_counter() - started_at
    result["model"] = data.get("model", model)
    return result
