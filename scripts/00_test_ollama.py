import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.ollama_client import generate_text


def main() -> None:
    prompt = "Кратко объясни, что такое RAG. Ответь на русском."
    result = generate_text(prompt)

    print(f"model: {result['model']}")
    print(f"latency_sec: {result['latency_sec']:.3f}")

    if result["error"]:
        print(f"error: {result['error']}")
        raise SystemExit(1)

    print("response:")
    print(result["text"])


if __name__ == "__main__":
    main()
