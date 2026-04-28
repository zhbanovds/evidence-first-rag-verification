import json
import re

from src.config import OLLAMA_TIMEOUT_SEC
from src.llm.ollama_client import generate_text
from src.prompts import ANSWER_FROM_FACT_PLAN_PROMPT, ANSWER_FROM_FACT_PLAN_STRICT_PROMPT


INSUFFICIENT_INFORMATION_ANSWER = (
    "В предоставленных источниках нет достаточной информации для ответа."
)

LIST_QUESTION_MARKERS = (
    "какие",
    "перечисли",
    "назови",
    "основные",
    "виды",
    "условия",
    "документы",
)

COMPARISON_QUESTION_MARKERS = (
    "чем отличается",
    "чем отличаются",
    "в чем отличие",
    "в чём отличие",
    "в чем разница",
    "в чём разница",
    "сравни",
)

NEGATION_MARKERS = (
    "не гарант",
    "не является",
    "не автоматически",
    "не всем",
    "не дает",
    "не даёт",
)

CONFLICT_MARKERS = (
    "противореч",
    "расхожд",
    "разные формулировки",
    "источники расходятся",
)

TOKEN_STOPWORDS = {
    "в", "во", "на", "по", "для", "и", "или", "а", "но", "что", "это",
    "как", "если", "при", "от", "до", "из", "за", "с", "со", "к", "ко",
    "не", "нет", "он", "она", "они", "его", "ее", "её", "их", "где", "у",
    "быть", "является", "являются", "может", "могут", "нужно",
}


def _has_possible_conflict(fact_plan) -> bool:
    return any(item.get("possible_conflict") is True for item in fact_plan)


def _mentions_conflict(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in CONFLICT_MARKERS)


def _ensure_conflict_mentioned(answer: str, fact_plan) -> str:
    if not _has_possible_conflict(fact_plan) or _mentions_conflict(answer):
        return answer

    return (
        answer.rstrip()
        + " В источниках также отмечено расхождение формулировок по вопросу "
        "гарантии или автоматического предоставления места."
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().replace("ё", "е").split())


def _is_list_question(question: str) -> bool:
    lowered = _normalize_text(question)
    return any(marker in lowered for marker in LIST_QUESTION_MARKERS)


def _is_comparison_question(question: str) -> bool:
    lowered = _normalize_text(question)
    return any(marker in lowered for marker in COMPARISON_QUESTION_MARKERS)


def _important_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-zа-я0-9]+", _normalize_text(text)):
        if token in TOKEN_STOPWORDS:
            continue
        if token.isdigit() or len(token) > 4:
            tokens.append(token)
    return tokens


def _numeric_tokens(text: str) -> list[str]:
    normalized = _normalize_text(text).replace(",", ".")
    return re.findall(r"\d+(?:\.\d+)?", normalized)


def _quoted_phrases(text: str) -> list[str]:
    return [
        phrase.strip()
        for phrase in re.findall(r"«([^»]+)»", str(text or ""))
        if phrase.strip()
    ]


def _tokens_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.isdigit() or right.isdigit():
        return False
    shortest = min(len(left), len(right))
    return shortest >= 5 and left[:5] == right[:5]


def _token_overlap_ratio(left_tokens: list[str], right_tokens: list[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0

    matched = 0
    for left_token in left_tokens:
        if any(_tokens_match(left_token, right_token) for right_token in right_tokens):
            matched += 1
    return matched / len(left_tokens)


def _phrase_is_used_in_answer(phrase: str, answer: str) -> bool:
    phrase_tokens = _important_tokens(phrase)
    answer_tokens = _important_tokens(answer)
    if not phrase_tokens:
        return True
    return _token_overlap_ratio(phrase_tokens, answer_tokens) >= 0.7


def _has_required_negation(claim: str, answer: str) -> bool:
    normalized_claim = _normalize_text(claim)
    if not any(marker in normalized_claim for marker in NEGATION_MARKERS):
        return True
    normalized_answer = _normalize_text(answer)
    return any(marker in normalized_answer for marker in NEGATION_MARKERS)


def _claim_is_used_in_answer(claim: str, answer: str, question: str = "") -> bool:
    normalized_claim = _normalize_text(claim)
    normalized_answer = _normalize_text(answer)
    if normalized_claim and normalized_claim in normalized_answer:
        return True

    claim_tokens = _important_tokens(claim)
    if not claim_tokens:
        return False

    answer_tokens = _important_tokens(answer)
    if not answer_tokens:
        return False

    claim_numbers = _numeric_tokens(claim)
    answer_numbers = set(_numeric_tokens(answer))
    if claim_numbers and not all(number in answer_numbers for number in claim_numbers):
        return False

    for phrase in _quoted_phrases(claim):
        if not _phrase_is_used_in_answer(phrase, answer):
            return False

    if not _has_required_negation(claim, answer):
        return False

    non_numeric_tokens = [token for token in claim_tokens if not token.isdigit()]
    non_numeric_answer_tokens = [token for token in answer_tokens if not token.isdigit()]
    overlap = _token_overlap_ratio(claim_tokens, answer_tokens)
    entity_overlap = _token_overlap_ratio(non_numeric_tokens, non_numeric_answer_tokens)

    if claim_numbers:
        return entity_overlap >= 0.3 or overlap >= 0.5

    if _is_comparison_question(question) or _is_comparison_question(claim):
        return overlap >= 0.5

    if len(claim_tokens) <= 3:
        return overlap >= 0.7

    if len(claim_tokens) <= 6:
        return overlap >= 0.55

    return overlap >= 0.45


def _used_claim_count(answer: str, fact_plan: list[dict]) -> int:
    return sum(
        1
        for item in fact_plan
        if _claim_is_used_in_answer(str(item.get("claim", "")), answer)
    )


def _is_checkable_fact_plan_item(item: dict) -> bool:
    claim = str(item.get("claim", "")).strip()
    if not claim:
        return False

    status = str(item.get("status", "")).lower().strip()
    if status in {"not_supported", "no_evidence", "unsupported"}:
        return False

    return True


def check_fact_plan_usage(question: str, fact_plan: list[dict], answer: str) -> dict:
    checkable_items = [
        item for item in fact_plan if _is_checkable_fact_plan_item(item)
    ]
    missing_items = []
    for item in checkable_items:
        claim = str(item.get("claim", ""))
        if not _claim_is_used_in_answer(claim, answer, question=question):
            missing_items.append(item)

    total = len(checkable_items)
    used = total - len(missing_items)
    rate = used / total if total else 0.0

    missing_claim_ids = [
        str(item.get("claim_id") or f"P{index:03d}")
        for index, item in enumerate(missing_items, start=1)
    ]
    missing_claims = [str(item.get("claim", "")) for item in missing_items]

    if total == 0:
        passed = True
        reason = "fact plan is empty"
    elif not str(answer or "").strip():
        passed = False
        reason = "answer is empty"
    elif missing_items:
        passed = False
        reason = (
            f"missing {len(missing_items)} of {total} fact plan claims: "
            + ", ".join(missing_claim_ids)
        )
    else:
        passed = True
        reason = "all fact plan claims are expressed in the answer"

    return {
        "total_fact_plan_claims": total,
        "used_claim_count": used,
        "missing_claim_ids": missing_claim_ids,
        "missing_claims": missing_claims,
        "fact_plan_used_in_answer_rate": rate,
        "answer_coverage_gate_passed": passed,
        "answer_coverage_gate_reason": reason,
    }


def _format_coverage_feedback(coverage_feedback) -> str:
    if not coverage_feedback:
        return "Предыдущий ответ не прошёл проверку полноты переноса fact plan."

    if isinstance(coverage_feedback, str):
        return coverage_feedback

    reason = coverage_feedback.get("answer_coverage_gate_reason", "")
    missing_claims = coverage_feedback.get("missing_claims", [])
    missing_ids = coverage_feedback.get("missing_claim_ids", [])
    lines = [reason or "Предыдущий ответ пропустил часть fact plan."]
    for claim_id, claim in zip(missing_ids, missing_claims):
        lines.append(f"- {claim_id}: {claim}")
    return "\n".join(lines)


def _is_too_short_for_fact_plan(answer: str, fact_plan: list[dict]) -> bool:
    if len(fact_plan) < 2:
        return False

    claims_text_length = sum(
        len(str(item.get("claim", "")).strip())
        for item in fact_plan
        if item.get("claim")
    )
    if claims_text_length <= 0:
        return False

    min_answer_length = max(90, int(claims_text_length * 0.35))
    return len(answer.strip()) < min_answer_length


def _needs_fallback_answer(answer: str, fact_plan: list[dict]) -> tuple[bool, list[str]]:
    reasons = []
    stripped_answer = answer.strip()
    lowered_answer = _normalize_text(stripped_answer)

    if not fact_plan:
        return False, reasons

    if not stripped_answer:
        reasons.append("empty_answer")

    if "нет достаточной информации" in lowered_answer:
        reasons.append("insufficient_information_answer")

    used_claims = _used_claim_count(stripped_answer, fact_plan)
    if len(fact_plan) >= 2 and used_claims < (len(fact_plan) / 2):
        reasons.append(
            f"used_less_than_half_claims:{used_claims}/{len(fact_plan)}"
        )

    if _is_too_short_for_fact_plan(stripped_answer, fact_plan):
        reasons.append("answer_too_short_for_fact_plan")

    return bool(reasons), reasons


def _claim_sentence(claim: str) -> str:
    claim = " ".join(str(claim or "").split()).strip()
    if not claim:
        return ""
    if claim.endswith((".", "!", "?")):
        return claim
    return claim + "."


def _build_fallback_answer(question: str, fact_plan: list[dict]) -> str:
    claim_sentences = [
        _claim_sentence(item.get("claim", ""))
        for item in fact_plan
        if str(item.get("claim", "")).strip()
    ]
    claim_sentences = [claim for claim in claim_sentences if claim]

    if not claim_sentences:
        return INSUFFICIENT_INFORMATION_ANSWER

    if _is_list_question(question):
        lines = ["Подтверждены следующие пункты:"]
        lines.extend([f"- {claim}" for claim in claim_sentences])
        answer = "\n".join(lines)
    elif _is_comparison_question(question):
        answer = " ".join(claim_sentences)
    else:
        answer = " ".join(claim_sentences)

    if _has_possible_conflict(fact_plan) and not _mentions_conflict(answer):
        answer = (
            answer.rstrip()
            + " В fact plan есть признак возможного расхождения источников, "
            "поэтому эти сведения следует трактовать осторожно."
        )

    return answer


def _call_llm(prompt: str, llm_client=None) -> dict:
    if llm_client is None:
        return generate_text(prompt, timeout_sec=OLLAMA_TIMEOUT_SEC, num_predict=384)

    if callable(llm_client):
        response = llm_client(prompt)
    elif hasattr(llm_client, "generate_text"):
        response = llm_client.generate_text(prompt)
    else:
        raise TypeError("llm_client must be callable or have generate_text(prompt).")

    if isinstance(response, dict):
        return response

    return {
        "text": str(response),
        "latency_sec": 0.0,
        "error": None,
    }


def generate_answer_from_fact_plan(
    question,
    fact_plan,
    llm_client=None,
    strict: bool = False,
    coverage_feedback=None,
) -> dict:
    result = {
        "answer": "",
        "raw_output": "",
        "error": None,
        "latency_sec": 0.0,
        "num_llm_calls": 0,
        "fallback_used": False,
        "fallback_reasons": [],
    }

    if not fact_plan:
        result["answer"] = INSUFFICIENT_INFORMATION_ANSWER
        result["raw_output"] = INSUFFICIENT_INFORMATION_ANSWER
        return result

    if strict:
        prompt = ANSWER_FROM_FACT_PLAN_STRICT_PROMPT.format(
            question=question,
            fact_plan=json.dumps(fact_plan, ensure_ascii=False, indent=2),
            coverage_feedback=_format_coverage_feedback(coverage_feedback),
        )
    else:
        prompt = ANSWER_FROM_FACT_PLAN_PROMPT.format(
            question=question,
            fact_plan=json.dumps(fact_plan, ensure_ascii=False, indent=2),
        )

    result["num_llm_calls"] = 1
    try:
        llm_result = _call_llm(prompt, llm_client=llm_client)
    except Exception as exc:
        result["error"] = f"Answer from fact plan LLM call failed: {exc}"
        return result

    raw_output = llm_result.get("text", "")
    result["raw_output"] = raw_output
    result["latency_sec"] = llm_result.get("latency_sec", 0.0)

    if llm_result.get("error"):
        result["error"] = llm_result["error"]
        return result

    answer = _ensure_conflict_mentioned(raw_output.strip(), fact_plan)
    needs_fallback, fallback_reasons = _needs_fallback_answer(answer, fact_plan)
    if needs_fallback:
        answer = _build_fallback_answer(question, fact_plan)
        result["fallback_used"] = True
        result["fallback_reasons"] = fallback_reasons

    result["answer"] = answer
    return result
