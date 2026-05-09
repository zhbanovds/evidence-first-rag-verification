import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    try:
        from src.library import VerifiedRAGSystem
        from src.schemas import RAGRequest
    except Exception as exc:
        print("[library_api] Cannot import library facade.")
        print(f"[library_api] Error: {exc}")
        print("[library_api] Check that project dependencies are installed.")
        return 0

    request = RAGRequest(
        question="Когда начинается приём документов в МАИ?",
        mode="clean",
        top_k=3,
    )

    try:
        rag = VerifiedRAGSystem(mode="clean")
        response = rag.ask(request)
    except Exception as exc:
        print("[library_api] Smoke-test could not run.")
        print(f"[library_api] Error: {exc}")
        print("[library_api] Check Chroma index and Ollama availability.")
        return 0

    print("[library_api] question:")
    print(response.question)
    print()
    print("[library_api] final answer:")
    print(response.final_answer or response.answer or "<empty>")
    print()
    print(f"[library_api] retrieved_chunks: {len(response.retrieved_chunks)}")
    print(f"[library_api] claims: {len(response.claims)}")
    print(f"[library_api] verification_report: {len(response.verification_report)}")
    print(f"[library_api] latency_sec: {response.latency_sec:.2f}")
    print(f"[library_api] num_llm_calls: {response.num_llm_calls}")

    if response.errors:
        print()
        print("[library_api] completed with errors:")
        for error in response.errors:
            print(f"- {error}")
        print()
        print("[library_api] Useful checks:")
        print("- python3 scripts/02_build_index.py --mode clean")
        print("- python3 scripts/00_test_ollama.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
