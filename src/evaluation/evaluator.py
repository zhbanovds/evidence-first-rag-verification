import csv
import json
from pathlib import Path

from src.config import EXPECTED_CLAIMS_PATH, RESULTS_CONFLICT_DIR, RESULTS_DIR, RESULTS_MAIN_DIR
from src.evaluation.metrics import summarize_results


SUMMARY_COLUMNS = [
    "mode",
    "pipeline",
    "total_questions",
    "support_rate",
    "unsupported_rate",
    "partial_support_rate",
    "conflict_rate",
    "no_evidence_rate",
    "required_claim_coverage",
    "forbidden_claim_rate",
    "expected_doc_hit_rate",
    "avg_fact_plan_claim_count",
    "fact_plan_evidence_coverage",
    "conflict_detection_rate",
    "wrong_conflict_resolution_rate",
    "avg_latency_sec",
    "avg_llm_calls",
    "avg_retrieval_calls",
    "avg_answer_length",
]

RESULT_FILES = {
    "main": {
        "baseline": "baseline_results.jsonl",
        "advanced": "advanced_results.jsonl",
        "posthoc": "posthoc_results.jsonl",
        "final_verified": "final_verified_results.jsonl",
    },
    "conflict": {
        "baseline": "baseline_results.jsonl",
        "advanced": "advanced_results.jsonl",
        "posthoc": "posthoc_results.jsonl",
        "final_verified": "final_verified_results.jsonl",
    },
}


def load_expected_claims(path: Path = EXPECTED_CLAIMS_PATH) -> dict:
    if not path.exists():
        return {"questions": {}}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def result_path(mode: str, filename: str) -> Path:
    if mode == "main":
        return RESULTS_MAIN_DIR / filename
    if mode == "conflict":
        return RESULTS_CONFLICT_DIR / filename
    raise ValueError("mode must be 'main' or 'conflict'")


def build_metrics_summary(expected_claims: dict) -> list[dict]:
    summaries = []
    for mode, pipeline_files in RESULT_FILES.items():
        for pipeline, filename in pipeline_files.items():
            path = result_path(mode, filename)
            if not path.exists():
                continue

            results = read_jsonl(path)
            if not results:
                continue

            summaries.append(
                summarize_results(
                    mode=mode,
                    pipeline=pipeline,
                    results=results,
                    expected_claims=expected_claims,
                )
            )

    return summaries


def write_metrics_summary(rows: list[dict], path: Path = RESULTS_DIR / "metrics_summary.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, 0.0) for column in SUMMARY_COLUMNS})


def _doc_ids(result: dict) -> list[str]:
    return [chunk.get("doc_id", "") for chunk in result.get("retrieved_chunks", [])]


def _verdicts(result: dict) -> list[str]:
    verdicts = []
    for item in result.get("verification_report", []):
        claim_id = item.get("claim_id", "")
        verdict = item.get("verdict", "")
        verdicts.append(f"{claim_id}: {verdict}" if claim_id else str(verdict))
    return verdicts


def _first_existing_result(mode: str, pipeline: str):
    filename = RESULT_FILES[mode][pipeline]
    path = result_path(mode, filename)
    rows = read_jsonl(path)
    if not rows:
        return None, None
    return path, rows[0]


def build_examples_for_thesis(path: Path = RESULTS_DIR / "examples_for_thesis.md") -> None:
    examples = [
        ("Baseline", "main", "baseline"),
        ("Advanced Retrieval", "main", "advanced"),
        ("Post-hoc Verification", "main", "posthoc"),
        ("Final Evidence-First", "main", "final_verified"),
        ("Conflict Example", "conflict", "final_verified"),
    ]

    sections = ["# Examples for Thesis", ""]
    for title, mode, pipeline in examples:
        source_path, result = _first_existing_result(mode, pipeline)
        if not result:
            continue

        answer = result.get("final_answer") or result.get("answer", "")
        sections.extend(
            [
                f"## {title}",
                "",
                f"**Source file:** `{source_path}`",
                "",
                f"**Question:** {result.get('question', '')}",
                "",
                "**Retrieved doc_id:**",
                "",
            ]
        )
        sections.extend([f"- `{doc_id}`" for doc_id in _doc_ids(result)])
        sections.extend(
            [
                "",
                "**Answer / Final answer:**",
                "",
                answer,
                "",
                "**Verification verdicts:**",
                "",
            ]
        )
        verdicts = _verdicts(result)
        if verdicts:
            sections.extend([f"- {verdict}" for verdict in verdicts])
        else:
            sections.append("- no verification report")
        sections.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")


def run_evaluation() -> list[dict]:
    expected_claims = load_expected_claims()
    rows = build_metrics_summary(expected_claims)
    write_metrics_summary(rows)
    build_examples_for_thesis()
    return rows
