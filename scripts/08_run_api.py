import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Alternative command:
# uvicorn src.api.main:app --host 127.0.0.1 --port 8000
def main() -> int:
    try:
        import uvicorn
    except Exception as exc:
        print("[run_api] uvicorn is not available.")
        print(f"[run_api] Error: {exc}")
        print("[run_api] Install dependencies with: python3 -m pip install -r requirements.txt")
        return 1

    try:
        from src.api.main import app
    except Exception as exc:
        print("[run_api] FastAPI application could not be imported.")
        print(f"[run_api] Error: {exc}")
        print("[run_api] Install dependencies with: python3 -m pip install -r requirements.txt")
        return 1

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
