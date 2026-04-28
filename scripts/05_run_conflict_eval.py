import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QUESTIONS_CONFLICT_PATH, RESULTS_CONFLICT_DIR
from src.pipelines.advanced_rag import run_advanced_rag
from src.pipelines.baseline_rag import run_baseline_rag
from src.pipelines.final_evidence_first_rag import run_final_evidence_first_rag
from src.pipelines.posthoc_verification_rag import run_posthoc_verification_rag


MODE = "conflict"
CONFLICT_TOP_K = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conflict evaluation pipelines.")
    parser.add_argument(
        "--pipeline",
        choices=[
            "baseline",
            "advanced",
            "posthoc",
            "final_verified",
            "final_evidence_first",
        ],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing question_id results instead of overwriting the output file.",
    )
    return parser.parse_args()


def load_questions(limit: Optional[int] = None) -> list[dict]:
    if not QUESTIONS_CONFLICT_PATH.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_CONFLICT_PATH}")

    with QUESTIONS_CONFLICT_PATH.open("r", encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        questions = questions[:limit]

    return questions


def get_output_path(pipeline: str) -> Path:
    filenames = {
        "baseline": "baseline_results.jsonl",
        "advanced": "advanced_results.jsonl",
        "posthoc": "posthoc_results.jsonl",
        "final_verified": "final_verified_results.jsonl",
        "final_evidence_first": "final_verified_results.jsonl",
    }
    return RESULTS_CONFLICT_DIR / filenames[pipeline]


def load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []

    results = []
    with output_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[conflict_eval] warning: skipped invalid JSON line {line_no}: {exc}",
                    flush=True,
                )
                continue
            if isinstance(parsed, dict):
                results.append(parsed)
    return results


def is_successful_result(result: dict) -> bool:
    errors = result.get("errors", [])
    has_errors = bool(errors)
    answer = str(result.get("answer") or "").strip()
    final_answer = str(result.get("final_answer") or "").strip()
    return not has_errors and bool(answer or final_answer)


def rewrite_successful_results(
    output_path: Path,
    existing_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    successful_results = []
    failed_results = []
    seen_question_ids = set()

    for result in existing_results:
        question_id = str(result.get("question_id", "")).strip()
        if question_id and is_successful_result(result) and question_id not in seen_question_ids:
            successful_results.append(result)
            seen_question_ids.add(question_id)
        else:
            failed_results.append(result)

    temp_path = output_path.with_name(output_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for result in successful_results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)
    return successful_results, failed_results


def run_single_question(
    pipeline: str,
    question_id: str,
    question: str,
) -> dict:
    if pipeline == "baseline":
        return run_baseline_rag(
            question_id=question_id,
            question=question,
            mode=MODE,
            top_k=CONFLICT_TOP_K,
        )
    if pipeline == "advanced":
        return run_advanced_rag(
            question_id=question_id,
            question=question,
            mode=MODE,
            top_k=CONFLICT_TOP_K,
        )
    if pipeline == "posthoc":
        return run_posthoc_verification_rag(
            question_id=question_id,
            question=question,
            mode=MODE,
            top_k=CONFLICT_TOP_K,
        )
    if pipeline in {"final_verified", "final_evidence_first"}:
        return run_final_evidence_first_rag(
            question_id=question_id,
            question=question,
            mode=MODE,
            top_k=CONFLICT_TOP_K,
        )
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def run_pipeline(
    args: argparse.Namespace,
    questions: list[dict],
    output_path: Path,
) -> list[dict]:
    existing_results = load_existing_results(output_path) if args.resume else []
    failed_results = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        results, failed_results = rewrite_successful_results(
            output_path,
            existing_results,
        )
    else:
        results = []

    completed_question_ids = {
        str(result.get("question_id", "")).strip()
        for result in results
        if result.get("question_id")
    }
    total = len(questions)
    new_results_count = 0

    open_mode = "a" if args.resume else "w"
    with output_path.open(open_mode, encoding="utf-8") as file:
        for index, row in enumerate(questions, start=1):
            question_id = row.get("question_id") or row.get("id") or f"CQ{index:03d}"
            question = row.get("question", "").strip()
            if args.resume and question_id in completed_question_ids:
                print(
                    f"[conflict_eval] {index}/{total} {args.pipeline} {question_id} skipped",
                    flush=True,
                )
                continue

            print(
                f"[conflict_eval] {index}/{total} {args.pipeline} {question_id}",
                flush=True,
            )
            result = run_single_question(
                pipeline=args.pipeline,
                question_id=question_id,
                question=question,
            )
            results.append(result)
            completed_question_ids.add(question_id)
            new_results_count += 1
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

    if args.resume:
        failed_question_ids = [
            str(result.get("question_id", "")).strip()
            for result in failed_results
            if result.get("question_id")
        ]
        print(f"[conflict_eval] resume existing: {len(existing_results)}")
        print(f"[conflict_eval] resume kept successful: {len(results) - new_results_count}")
        print(f"[conflict_eval] resume removed failed: {len(failed_results)}")
        print(f"[conflict_eval] resume removed question_ids: {failed_question_ids}")
        print(f"[conflict_eval] resume appended: {new_results_count}")

    return results


def main() -> None:
    args = parse_args()
    questions = load_questions(args.limit)
    output_path = get_output_path(args.pipeline)

    print(f"[conflict_eval] pipeline: {args.pipeline}")
    print(f"[conflict_eval] mode: {MODE}")
    print(f"[conflict_eval] conflict evaluation top_k={CONFLICT_TOP_K}")
    print(f"[conflict_eval] questions: {len(questions)}")
    print(f"[conflict_eval] output: {output_path}")
    print(f"[conflict_eval] resume: {args.resume}")

    results = run_pipeline(args, questions, output_path)

    errors_count = sum(1 for result in results if result["errors"])
    print(f"[conflict_eval] saved: {len(results)}")
    print(f"[conflict_eval] results with errors: {errors_count}")


if __name__ == "__main__":
    main()
