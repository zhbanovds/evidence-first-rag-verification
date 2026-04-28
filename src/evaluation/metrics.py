import re
from statistics import mean


VERDICTS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "NOT_SUPPORTED",
    "CONFLICT",
    "NO_EVIDENCE",
}

RUSSIAN_STOPWORDS = {
    "в", "во", "на", "по", "для", "и", "или", "а", "но", "что", "это",
    "как", "если", "при", "от", "до", "из", "за", "с", "со", "к", "ко",
    "не", "нет", "он", "она", "они", "его", "ее", "их", "где", "у",
    "быть", "является", "являются", "может", "могут", "нужно",
}

CONFLICT_MARKERS = (
    "противореч",
    "расхожд",
    "разные сведения",
    "разные данные",
    "источники расходятся",
)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().replace("ё", "е").split())


def _tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-zа-я0-9]+", normalized)


def _important_tokens(text: str) -> list[str]:
    tokens = []
    for token in _tokens(text):
        if token in RUSSIAN_STOPWORDS:
            continue
        if token.isdigit() or len(token) > 4:
            tokens.append(token)
    return tokens


def _tokens_match(expected_token: str, answer_token: str) -> bool:
    if expected_token == answer_token:
        return True
    if expected_token.isdigit() or answer_token.isdigit():
        return False

    min_prefix = 5
    shortest = min(len(expected_token), len(answer_token))
    if shortest < min_prefix:
        return False

    return expected_token[:min_prefix] == answer_token[:min_prefix]


def required_claim_matches_text(claim: str, text: str, threshold: float = 0.55) -> bool:
    claim_tokens = _important_tokens(claim)
    if not claim_tokens:
        return False

    text_tokens = _important_tokens(text)
    if not text_tokens:
        return False

    matched = 0
    for claim_token in claim_tokens:
        if any(_tokens_match(claim_token, text_token) for text_token in text_tokens):
            matched += 1

    return safe_divide(matched, len(claim_tokens)) >= threshold


def forbidden_claim_matches_text(claim: str, text: str) -> bool:
    normalized_claim = normalize_text(claim)
    normalized_text = normalize_text(text)
    return bool(normalized_claim and normalized_claim in normalized_text)


def phrase_matches_text(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    normalized_text = normalize_text(text)
    if not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True

    phrase_tokens = _important_tokens(phrase)
    if not phrase_tokens:
        return False

    text_tokens = set(_important_tokens(text))
    numeric_tokens = [token for token in phrase_tokens if token.isdigit()]
    if numeric_tokens and not all(token in text_tokens for token in numeric_tokens):
        return False

    matched = sum(1 for token in phrase_tokens if token in text_tokens)
    threshold = 1.0 if len(phrase_tokens) <= 2 else 0.65
    return safe_divide(matched, len(phrase_tokens)) >= threshold


def answer_text(result: dict) -> str:
    return result.get("final_answer") or result.get("answer") or ""


def get_expected(expected_claims: dict, question_id: str) -> dict:
    questions = expected_claims.get("questions", {})
    return questions.get(question_id, {})


def _normalize_doc_id(value: str) -> str:
    value = str(value or "").split("/")[-1]
    if value.endswith(".md"):
        value = value[:-3]
    return value


def _retrieved_doc_ids(result: dict) -> set[str]:
    doc_ids = set()
    for chunk in result.get("retrieved_chunks", []):
        doc_id = chunk.get("doc_id")
        source_path = chunk.get("source_path")
        if doc_id:
            doc_ids.add(_normalize_doc_id(doc_id))
        if source_path:
            doc_ids.add(_normalize_doc_id(source_path))
    return doc_ids


def verdict_rates(results: list[dict]) -> dict:
    counts = {verdict: 0 for verdict in VERDICTS}
    total = 0

    for result in results:
        for item in result.get("verification_report", []):
            verdict = str(item.get("verdict", "")).upper()
            if verdict in counts:
                counts[verdict] += 1
                total += 1

    return {
        "support_rate": safe_divide(counts["SUPPORTED"], total),
        "unsupported_rate": safe_divide(counts["NOT_SUPPORTED"], total),
        "partial_support_rate": safe_divide(counts["PARTIALLY_SUPPORTED"], total),
        "conflict_rate": safe_divide(counts["CONFLICT"], total),
        "no_evidence_rate": safe_divide(counts["NO_EVIDENCE"], total),
    }


def metadata_metrics(results: list[dict]) -> dict:
    return {
        "avg_latency_sec": mean([r.get("latency_sec", 0.0) for r in results]) if results else 0.0,
        "avg_llm_calls": mean([r.get("num_llm_calls", 0) for r in results]) if results else 0.0,
        "avg_retrieval_calls": mean([r.get("num_retrieval_calls", 0) for r in results]) if results else 0.0,
        "avg_answer_length": mean([len(answer_text(r)) for r in results]) if results else 0.0,
    }


def expected_claim_metrics(results: list[dict], expected_claims: dict) -> dict:
    required_scores = []
    forbidden_total = 0
    forbidden_hits = 0
    expected_doc_hits = []

    for result in results:
        expected = get_expected(expected_claims, result.get("question_id"))
        text = answer_text(result)

        required_claims = expected.get("required_claims", [])
        if required_claims:
            matched_required = sum(
                1 for claim in required_claims if required_claim_matches_text(claim, text)
            )
            required_scores.append(safe_divide(matched_required, len(required_claims)))

        forbidden_claims = expected.get("forbidden_claims", [])
        forbidden_total += len(forbidden_claims)
        forbidden_hits += sum(
            1 for claim in forbidden_claims if forbidden_claim_matches_text(claim, text)
        )

        expected_docs = expected.get("expected_docs", [])
        if expected_docs:
            expected_doc_ids = {_normalize_doc_id(doc) for doc in expected_docs}
            retrieved_doc_ids = _retrieved_doc_ids(result)
            expected_doc_hits.append(
                1.0 if expected_doc_ids.intersection(retrieved_doc_ids) else 0.0
            )

    return {
        "required_claim_coverage": mean(required_scores) if required_scores else 0.0,
        "forbidden_claim_rate": safe_divide(forbidden_hits, forbidden_total),
        "expected_doc_hit_rate": mean(expected_doc_hits) if expected_doc_hits else 0.0,
    }


def fact_plan_metrics(results: list[dict]) -> dict:
    fact_plan_counts = []
    evidence_coverages = []

    for result in results:
        fact_plan = result.get("fact_plan", [])
        fact_plan_counts.append(len(fact_plan))
        if fact_plan:
            with_evidence = sum(
                1 for item in fact_plan if item.get("evidence_chunk_ids")
            )
            evidence_coverages.append(safe_divide(with_evidence, len(fact_plan)))

    return {
        "avg_fact_plan_claim_count": mean(fact_plan_counts) if fact_plan_counts else 0.0,
        "fact_plan_evidence_coverage": mean(evidence_coverages) if evidence_coverages else 0.0,
    }


def _has_conflict_detection(result: dict) -> bool:
    for item in result.get("verification_report", []):
        if str(item.get("verdict", "")).upper() == "CONFLICT":
            return True

    for item in result.get("fact_plan", []):
        if item.get("possible_conflict") is True:
            return True

    final_answer = normalize_text(answer_text(result))
    return any(marker in final_answer for marker in CONFLICT_MARKERS)


def conflict_metrics(results: list[dict], expected_claims: dict, mode: str) -> dict:
    if mode != "conflict":
        return {
            "conflict_detection_rate": 0.0,
            "wrong_conflict_resolution_rate": 0.0,
        }

    if not results:
        return {
            "conflict_detection_rate": 0.0,
            "wrong_conflict_resolution_rate": 0.0,
        }

    detected = sum(1 for result in results if _has_conflict_detection(result))
    wrong = 0
    for result in results:
        expected = get_expected(expected_claims, result.get("question_id"))
        forbidden_claims = expected.get("forbidden_claims", [])
        text = answer_text(result)
        if any(forbidden_claim_matches_text(claim, text) for claim in forbidden_claims):
            wrong += 1

    return {
        "conflict_detection_rate": safe_divide(detected, len(results)),
        "wrong_conflict_resolution_rate": safe_divide(wrong, len(results)),
    }


def summarize_results(
    mode: str,
    pipeline: str,
    results: list[dict],
    expected_claims: dict,
) -> dict:
    summary = {
        "mode": mode,
        "pipeline": pipeline,
        "total_questions": len(results),
    }
    summary.update(verdict_rates(results))
    summary.update(expected_claim_metrics(results, expected_claims))
    summary.update(fact_plan_metrics(results))
    summary.update(conflict_metrics(results, expected_claims, mode))
    summary.update(metadata_metrics(results))
    return summary
