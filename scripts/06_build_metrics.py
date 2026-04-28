import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.evaluation.evaluator import run_evaluation


def main() -> None:
    rows = run_evaluation()
    print(f"[build_metrics] summaries: {len(rows)}")
    print(f"[build_metrics] metrics: {RESULTS_DIR / 'metrics_summary.csv'}")
    print(f"[build_metrics] examples: {RESULTS_DIR / 'examples_for_thesis.md'}")


if __name__ == "__main__":
    main()
