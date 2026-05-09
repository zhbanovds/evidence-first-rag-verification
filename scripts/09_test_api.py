import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = "http://127.0.0.1:8000"


def main() -> int:
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        print("API server is not running. Start it with: python3 scripts/08_run_api.py")
        return 0

    print(f"[api_test] GET /health status_code: {health_response.status_code}")
    try:
        print(f"[api_test] health: {health_response.json()}")
    except ValueError:
        print(f"[api_test] health response text: {health_response.text}")

    payload = {
        "question": "Когда начинается приём документов в МАИ?",
        "mode": "clean",
        "top_k": 3,
    }

    try:
        ask_response = requests.post(f"{API_URL}/ask", json=payload, timeout=900)
    except requests.exceptions.RequestException as exc:
        print(f"[api_test] POST /ask failed: {exc}")
        return 0

    print(f"[api_test] POST /ask status_code: {ask_response.status_code}")
    try:
        data = ask_response.json()
    except ValueError:
        print(f"[api_test] ask response text: {ask_response.text}")
        return 0

    answer = data.get("final_answer") or data.get("answer") or "<empty>"
    print("[api_test] answer:")
    print(answer)
    print(f"[api_test] retrieved_chunks: {len(data.get('retrieved_chunks') or [])}")

    errors = data.get("errors") or []
    if errors:
        print("[api_test] errors:")
        for error in errors:
            print(f"- {error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
