import json
import re
from functools import lru_cache
from typing import Optional

from src.config import FACT_PLAN_MAX_CLAIMS, OLLAMA_TIMEOUT_SEC, PROCESSED_DIR
from src.llm.ollama_client import generate_text
from src.prompts import FACT_PLAN_PROMPT
from src.utils import safe_json_loads


ALLOWED_CONFIDENCE = {"high", "medium", "low"}


NEGATIVE_GUARANTEE_MARKERS = (
    "не гарант",
    "не является автомат",
    "не равна гарант",
    "не даёт автоматического права",
    "не дает автоматического права",
    "не всем",
    "конкурс",
    "конкурсной основе",
    "при наличии мест",
    "при наличии свободных мест",
)

POSITIVE_PROVISION_MARKERS = (
    "предоставляется",
    "предоставляются",
    "получить могут",
    "имеют право",
    "гарантируется",
    "автоматически",
    "всем",
    "достаточным основанием",
)

LIST_QUESTION_MARKERS = (
    "какие",
    "перечисли",
    "назови",
    "основные",
    "виды",
    "условия",
    "документы",
    "особенности",
    "особенность",
)

COMPARISON_QUESTION_MARKERS = (
    "чем отличается",
    "чем отличаются",
    "в чем отличие",
    "в чём отличие",
    "отличаются",
)

NUMERIC_QUESTION_MARKERS = (
    "сколько",
    "баллы",
    "балл",
    "мест",
    "количество",
)

DEADLINE_QUESTION_MARKERS = (
    "когда",
    "срок",
    "до какого",
)

CONDITION_QUESTION_MARKERS = (
    "почему",
    "гарантирует",
    "гарантируется",
    "нужно ли",
    "можно ли",
)

INTENT_INSTRUCTIONS = {
    "list": (
        "- Это list-вопрос: создай отдельный claim на каждый найденный пункт списка, "
        "категорию, документ, условие или вариант. Не схлопывай список в один общий claim."
    ),
    "comparison": (
        "- Это comparison-вопрос: fact plan должен покрыть обе сравниваемые стороны. "
        "Добавь отдельные claims про первую сторону, вторую сторону и ключевые отличия, "
        "если они прямо есть в evidence."
    ),
    "numeric": (
        "- Это numeric-вопрос: если в evidence есть числа, каждый релевантный claim должен "
        "содержать и сущность, и число. Не пиши claim только про наличие показателя без значения."
    ),
    "deadline": (
        "- Это deadline-вопрос: каждый релевантный claim должен содержать дату или время "
        "и условие, к которому относится этот срок."
    ),
    "condition": (
        "- Это condition-вопрос: включи claims про правило и про ограничение, причину "
        "или условие применения правила."
    ),
    "default": (
        "- Это общий вопрос: выбери все центральные evidence-backed facts, которые нужны "
        "для полного ответа, без добавления внешних фактов."
    ),
}

LIST_INTRO_PATTERN = re.compile(
    r"(?:являются|выделяются|представлены|относятся|включают(?:ся)?|указаны|перечислены|должен\s+проверить)"
    r"\s*:?\s*(?P<items>[^.!?\n#]+)",
    flags=re.IGNORECASE,
)

BUDGET_PLACE_ITEM_PATTERN = re.compile(
    r"«(?P<name>[^»]+)»\s*[—-]\s*(?P<count>\d+)\s+(?:бюджетн\w*\s+)?мест\w*",
    flags=re.IGNORECASE,
)

PROGRAM_CODE_ITEM_PATTERN = re.compile(
    r"(?P<code>\d{2}\.\d{2}\.\d{2})\s+«(?P<name>[^»]+)»",
    flags=re.IGNORECASE,
)

SCORE_TABLE_PATTERN = re.compile(
    r"\|\s*(?P<subject>[А-Яа-яA-Za-zёЁ ]{4,80}?)\s*\|\s*(?P<score>\d{2,3})\s*\|"
)

SCORE_SENTENCE_PATTERN = re.compile(
    r"минимальн\w*\s+балл\s+по\s+(?P<subject>[а-яё ]{4,80}?)\s+составляет\s+(?P<score>\d{2,3})",
    flags=re.IGNORECASE,
)

MONEY_RANGE_PATTERN = re.compile(
    r"(?P<amount>\d{1,3}(?:\s?\d{3})*)\s+рубл[а-яё]*\s+[—-]\s+для\s+(?P<condition>[^.;\n]+)",
    flags=re.IGNORECASE,
)

MONEY_SENTENCE_PATTERN = re.compile(r"[^.!?]*\d{1,3}(?:\s?\d{3})\s+рубл[а-яё]*[^.!?]*[.!?]", flags=re.IGNORECASE)

BVI_STIPEND_PATTERN = re.compile(
    r"повышенн\w*\s+стипенд\w*\s+в\s+размере\s+"
    r"(?P<amount>\d{1,3}(?:\s?\d{3})*)\s+рубл[а-яё]*(?P<period>\s+в\s+месяц)?",
    flags=re.IGNORECASE,
)

ONE_TIME_PAYMENT_PATTERN = re.compile(
    r"единоразов\w*\s+выплат\w*[^.]*?абитуриентам,?\s*"
    r"(?P<condition>[^.]*?)\s*,\s*в\s+размере\s+"
    r"(?P<amount>\d{1,3}(?:\s?\d{3})*)\s+рубл[а-яё]*",
    flags=re.IGNORECASE,
)

DATE_LIKE_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}\s*)?\d{1,2}\s+[а-яё]+\s+\d{4}\s+года|\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}:\d{2}",
    flags=re.IGNORECASE,
)

QUESTION_STOPWORDS = {
    "как",
    "какие",
    "какой",
    "какая",
    "какое",
    "когда",
    "где",
    "что",
    "чем",
    "для",
    "или",
    "при",
    "про",
    "это",
    "нужно",
    "можно",
    "маи",
    "сайте",
    "поступающего",
    "поступающий",
    "поступлении",
    "поступления",
}

ENTITY_FOCUS_RULES = (
    {
        "entity_type": "admission_directions",
        "question_markers": ("направления поступления", "основные направления", "уровни поступления"),
        "focus_terms": (
            "направления поступления",
            "базовое высшее образование",
            "специализированное высшее образование",
            "среднее профессиональное образование",
            "иностранные граждане",
        ),
        "positive_doc_markers": ("admission_overview",),
        "positive_text_markers": (
            "основные уровни поступления",
            "основными блоками являются",
            "базовое высшее образование",
            "специализированное высшее образование",
            "среднее профессиональное образование",
            "поступление иностранных граждан",
        ),
        "negative_doc_markers": ("official_regulations",),
        "negative_text_markers": (
            "структура раздела «официальные документы»",
            "нормативные документы",
            "план приёма",
            "вступительные испытания, проводимые маи",
        ),
    },
    {
        "entity_type": "applicant_documents",
        "question_markers": ("какие документы", "документы представляет", "подаче заявления", "подача заявления"),
        "focus_terms": (
            "документ удостоверяющий личность",
            "документ установленного образца об образовании",
            "документ об образовании",
            "снилс",
            "индивидуальные достижения",
        ),
        "positive_doc_markers": ("application_documents",),
        "positive_text_markers": (
            "при подаче заявления",
            "поступающий представляет",
            "документ удостоверяющий личность",
            "документ установленного образца",
            "документ об образовании",
        ),
        "negative_doc_markers": ("official_regulations",),
        "negative_text_markers": (
            "раздел официальных документов",
            "нормативные документы",
            "план приёма",
            "вступительные испытания",
        ),
    },
    {
        "entity_type": "application_limits",
        "question_markers": ("сколько заявлений", "заявлений и направлений", "одно заявление"),
        "focus_terms": (
            "одно заявление",
            "не более пяти направлений",
            "направлений подготовки",
            "специальностей",
        ),
        "positive_doc_markers": ("application_documents",),
        "positive_text_markers": (
            "одно заявление",
            "до пяти направлений",
            "не более пяти направлений",
        ),
        "negative_doc_markers": (),
        "negative_text_markers": (),
    },
    {
        "entity_type": "target_obligations",
        "question_markers": ("обязательства", "целевом обучении", "целевое обучение"),
        "focus_terms": (
            "договор",
            "заказчик",
            "обязательства студента",
            "освоить образовательную программу",
            "отработать не менее трёх лет",
        ),
        "positive_doc_markers": ("target_admission",),
        "positive_text_markers": (
            "договор о целевом обучении",
            "обязательства студента",
            "освоить образовательную программу",
            "отработать не менее трёх лет",
            "заказчик",
        ),
        "negative_doc_markers": ("career_opportunities", "network_education"),
        "negative_text_markers": ("карьерные возможности", "сетевые программы"),
    },
    {
        "entity_type": "quotas",
        "question_markers": ("квота", "квоты", "особая", "отдельная", "целевая"),
        "focus_terms": (
            "особая квота",
            "отдельная квота",
            "целевая квота",
            "социальные и правовые категории",
            "часть 5.1 статьи 71",
            "договор о целевом обучении",
            "заказчик",
        ),
        "positive_doc_markers": ("admission_quotas", "special_rights"),
        "positive_text_markers": (
            "отличие целевой квоты",
            "особая квота",
            "отдельная квота",
            "целевая квота",
            "частью 5.1 статьи 71",
        ),
        "negative_doc_markers": ("ege_calculator",),
        "negative_text_markers": ("калькулятор",),
    },
    {
        "entity_type": "paid_vs_budget_plan",
        "question_markers": ("план приёма на платные", "план платных", "плана бюджетных", "бюджетных мест"),
        "focus_terms": (
            "план приёма на бюджетные места",
            "план приёма на платные места",
            "федерального бюджета",
            "средств физических",
            "юридических лиц",
            "количество бюджетных и платных мест",
        ),
        "positive_doc_markers": ("budget_admission_plan", "paid_admission_plan", "paid_admission"),
        "positive_text_markers": (
            "план приёма на бюджетные места",
            "план приёма на платные места",
            "средств федерального бюджета",
            "средств физических",
            "юридических лиц",
            "количество бюджетных и платных мест",
        ),
        "negative_doc_markers": (),
        "negative_text_markers": ("минимальные баллы", "минимальных баллов", "значения отличаются от минимальных"),
    },
    {
        "entity_type": "enrollment_list_order",
        "question_markers": ("конкурсный список", "конкурсный спис", "приказ о зачислении"),
        "focus_terms": (
            "конкурсный список",
            "приказ о зачислении",
            "сумма конкурсных баллов",
            "не является приказом",
            "факт зачисления",
        ),
        "positive_doc_markers": ("enrollment_orders",),
        "positive_text_markers": (
            "отличие приказа от конкурсного списка",
            "конкурсные списки",
            "не является приказом о зачислении",
            "факт зачисления",
        ),
        "negative_doc_markers": (),
        "negative_text_markers": ("вторая ошибка", "искать себя не в том приказе"),
    },
    {
        "entity_type": "network_programs",
        "question_markers": ("сетевые образовательные", "сетевая программа", "сетевые программы"),
        "focus_terms": (
            "сетевая программа",
            "нескольких организаций",
            "партнёр",
            "производственная база",
            "практики",
            "не отменяет конкурс",
            "вступительные испытания",
            "минимальные баллы",
        ),
        "positive_doc_markers": ("network_education_programs",),
        "positive_text_markers": (
            "главные отличия сетевой программы",
            "участие нескольких организаций",
            "производственной базой",
            "не отменяет конкурс",
        ),
        "negative_doc_markers": ("open_days",),
        "negative_text_markers": ("дни открытых дверей", "расписание мероприятий"),
    },
    {
        "entity_type": "career_opportunities",
        "question_markers": ("карьерные возможности", "индустриальные проекты", "работодател", "крылья ростеха"),
        "focus_terms": (
            "карьерные возможности",
            "работодатели",
            "индустриальные партнёры",
            "день карьеры",
            "один день",
            "крылья ростеха",
            "индустриальные проекты",
        ),
        "positive_doc_markers": ("career_opportunities",),
        "positive_text_markers": (
            "карьерные возможности",
            "работодателями",
            "индустриальные партнёры",
            "день карьеры",
            "один день",
            "крылья ростеха",
        ),
        "negative_doc_markers": ("features_of_basic_higher",),
        "negative_text_markers": ("базовое высшее образование",),
    },
    {
        "entity_type": "scores",
        "question_markers": ("минимальные баллы", "проходные баллы", "баллы", "балл"),
        "focus_terms": (
            "минимальные баллы",
            "проходные баллы",
            "нижний порог",
            "порог допуска",
            "конкурсная ситуация",
            "не гарантирует зачисление",
        ),
        "positive_doc_markers": ("minimum_scores", "ege_calculator"),
        "positive_text_markers": (
            "отличие минимальных баллов от проходных баллов",
            "минимальный балл",
            "проходной балл",
            "порог допуска",
        ),
        "negative_doc_markers": ("scholarships",),
        "negative_text_markers": ("стипенд", "рубл"),
    },
)


def _call_llm(prompt: str, llm_client=None) -> dict:
    if llm_client is None:
        return generate_text(prompt, timeout_sec=OLLAMA_TIMEOUT_SEC, num_predict=768)

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


def prepare_evidence_chunks(
    retrieved_chunks: list[dict],
    max_chunks_per_doc: int = 4,
) -> list[dict]:
    if max_chunks_per_doc <= 0:
        raise ValueError("max_chunks_per_doc must be positive")

    selected_chunks = []
    doc_counts = {}
    for chunk in retrieved_chunks:
        doc_id = chunk.get("doc_id")
        count = doc_counts.get(doc_id, 0)
        if count >= max_chunks_per_doc:
            continue

        doc_counts[doc_id] = count + 1
        selected_chunks.append(chunk)

    return selected_chunks


def classify_question_intent(question: str) -> str:
    lowered = _normalize_for_match(question)
    if any(marker in lowered for marker in COMPARISON_QUESTION_MARKERS):
        return "comparison"
    if "отлич" in lowered and ("чем" in lowered or "разниц" in lowered):
        return "comparison"
    if any(marker in lowered for marker in NUMERIC_QUESTION_MARKERS):
        return "numeric"
    if any(marker in lowered for marker in DEADLINE_QUESTION_MARKERS):
        return "deadline"
    if any(marker in lowered for marker in CONDITION_QUESTION_MARKERS):
        return "condition"
    if any(marker in lowered for marker in LIST_QUESTION_MARKERS):
        return "list"
    return "default"


def analyze_question_focus(question: str) -> dict:
    lowered = _normalize_for_match(question)
    intent = classify_question_intent(question)
    entity_type = _detect_expected_entity_type(lowered)
    rule = _focus_rule(entity_type)

    focus_terms = []
    if rule:
        focus_terms.extend(rule.get("focus_terms", ()))
    focus_terms.extend(_extract_question_phrases(lowered))

    comparison_sides = _extract_comparison_sides(lowered, entity_type)
    numeric_targets = _extract_numeric_targets(lowered, entity_type)

    for side in comparison_sides:
        focus_terms.append(side)
    for target in numeric_targets:
        focus_terms.append(target)

    return {
        "intent": intent,
        "question_focus_terms": _dedupe_strings(focus_terms),
        "comparison_sides": comparison_sides,
        "numeric_targets": numeric_targets,
        "expected_entity_type": entity_type,
    }


def _dedupe_strings(values) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        value = " ".join(str(value or "").split()).strip(" .,:;")
        if not value:
            continue
        key = _normalize_for_match(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _focus_rule(entity_type: str) -> Optional[dict]:
    for rule in ENTITY_FOCUS_RULES:
        if rule["entity_type"] == entity_type:
            return rule
    return None


def _detect_expected_entity_type(lowered_question: str) -> str:
    for rule in ENTITY_FOCUS_RULES:
        if any(marker in lowered_question for marker in rule.get("question_markers", ())):
            return str(rule["entity_type"])

    if "срок" in lowered_question or "когда" in lowered_question:
        return "deadlines"
    if "мест" in lowered_question:
        return "places"
    return "general"


def _extract_question_phrases(lowered_question: str) -> list[str]:
    phrases = []
    for match in re.finditer(
        r"(?:какие|какой|какая|чем|сколько|для чего|зачем)\s+([^?]{4,90})",
        lowered_question,
    ):
        phrase = match.group(1)
        phrase = re.split(r"\s+(?:в|на|при|для|и почему|$)", phrase, maxsplit=1)[0]
        if len(phrase) >= 4:
            phrases.append(phrase)

    for token in re.findall(r"[a-zа-я0-9]+", lowered_question):
        if token in QUESTION_STOPWORDS:
            continue
        if token.isdigit() or len(token) >= 5:
            phrases.append(token)
    return phrases


def _extract_comparison_sides(lowered_question: str, entity_type: str) -> list[str]:
    if entity_type == "scores" and "минималь" in lowered_question and "проходн" in lowered_question:
        return ["минимальные баллы", "проходные баллы"]
    if entity_type == "enrollment_list_order":
        return ["конкурсный список", "приказ о зачислении"]
    if entity_type == "quotas":
        sides = []
        if "особ" in lowered_question:
            sides.append("особая квота")
        if "отдельн" in lowered_question:
            sides.append("отдельная квота")
        if "целев" in lowered_question:
            sides.append("целевая квота")
        return sides
    if entity_type == "paid_vs_budget_plan":
        return ["план приёма на платные места", "план приёма на бюджетные места"]
    if entity_type == "network_programs":
        return ["сетевая программа", "обычная образовательная программа"]
    if "базов" in lowered_question and "специализирован" in lowered_question:
        return ["базовое высшее образование", "специализированное высшее образование"]
    if "вступительн" in lowered_question and "без" in lowered_question:
        return ["поступающие со вступительными испытаниями", "поступающие без вступительных испытаний"]

    match = re.search(r"чем\s+(.+?)\s+отлича\w*\s+от\s+(.+?)(?:\?|$)", lowered_question)
    if match:
        return _dedupe_strings([match.group(1), match.group(2)])
    return []


def _extract_numeric_targets(lowered_question: str, entity_type: str) -> list[str]:
    targets = []
    if entity_type == "application_limits":
        if "заявлен" in lowered_question:
            targets.append("заявление")
        if "направлен" in lowered_question or "специальност" in lowered_question:
            targets.append("направления подготовки")
        return targets

    if "балл" in lowered_question:
        targets.append("баллы")
        for subject in (
            "русский язык",
            "математика",
            "информатика",
            "физика",
            "обществознание",
            "иностранный язык",
        ):
            if subject in lowered_question:
                targets.append(subject)
    if "мест" in lowered_question:
        targets.append("места")
    if "заявлен" in lowered_question:
        targets.append("заявление")
    if "направлен" in lowered_question:
        targets.append("направления подготовки")
    return _dedupe_strings(targets)


def _infer_mode(retrieved_chunks: list[dict], mode: Optional[str] = None) -> str:
    if mode in {"clean", "conflict"}:
        return mode

    for chunk in retrieved_chunks:
        if str(chunk.get("mode", "")).strip() == "conflict":
            return "conflict"
        if chunk.get("is_conflict_source") is True:
            return "conflict"
        if "raw_conflict" in str(chunk.get("source_path", "")):
            return "conflict"
    return "clean"


@lru_cache(maxsize=2)
def _load_processed_chunks(mode: str) -> tuple[dict, dict]:
    path = PROCESSED_DIR / f"{mode}_chunks.jsonl"
    by_id = {}
    by_doc_index = {}
    if not path.exists():
        return by_id, by_doc_index

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id")
            doc_id = chunk.get("doc_id")
            chunk_index = chunk.get("chunk_index")
            if chunk_id:
                by_id[str(chunk_id)] = chunk
            if doc_id and chunk_index is not None:
                by_doc_index[(str(doc_id), int(chunk_index))] = chunk
    return by_id, by_doc_index


def _chunk_index(chunk: dict) -> Optional[int]:
    if chunk.get("chunk_index") is not None:
        try:
            return int(chunk["chunk_index"])
        except (TypeError, ValueError):
            pass

    chunk_id = str(chunk.get("chunk_id", ""))
    match = re.search(r"__(\d+)$", chunk_id)
    if match:
        return int(match.group(1))
    return None


def _minimal_chunk(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "text": chunk.get("text", ""),
        "chunk_index": _chunk_index(chunk),
    }


def expand_neighbor_chunks(
    chunks: list[dict],
    mode: str,
    radius: int = 1,
    max_total_chunks: int = 20,
) -> tuple[list[dict], bool]:
    if radius <= 0:
        return [_minimal_chunk(chunk) for chunk in chunks], False

    _, by_doc_index = _load_processed_chunks(mode)
    expanded = []
    seen = set()
    neighbor_added = False

    def add_chunk(chunk: dict, is_neighbor: bool = False) -> None:
        nonlocal neighbor_added
        if len(expanded) >= max_total_chunks:
            return
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            return
        seen.add(chunk_id)
        expanded.append(_minimal_chunk(chunk))
        if is_neighbor:
            neighbor_added = True

    for chunk in chunks:
        add_chunk(chunk)
        doc_id = chunk.get("doc_id")
        index = _chunk_index(chunk)
        if not doc_id or index is None:
            continue
        for neighbor_index in range(index - radius, index + radius + 1):
            if neighbor_index == index:
                continue
            neighbor = by_doc_index.get((str(doc_id), neighbor_index))
            if neighbor:
                add_chunk(neighbor, is_neighbor=True)

    return expanded, neighbor_added


def expand_query_matched_same_doc_chunks(
    chunks: list[dict],
    mode: str,
    question: str,
    max_total_chunks: int = 24,
    max_extra_per_doc: int = 6,
) -> tuple[list[dict], bool]:
    _, by_doc_index = _load_processed_chunks(mode)
    if not by_doc_index or len(chunks) >= max_total_chunks:
        return chunks, False

    query_tokens = _query_tokens(question)
    if not query_tokens:
        return chunks, False

    selected = list(chunks)
    seen = {str(chunk.get("chunk_id")) for chunk in selected if chunk.get("chunk_id")}
    doc_ids = []
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)

    added = False
    for doc_id in doc_ids:
        candidates = []
        for (candidate_doc_id, _), chunk in by_doc_index.items():
            if candidate_doc_id != doc_id:
                continue
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            lowered_text = _normalize_for_match(chunk.get("text", ""))
            text_tokens = re.findall(r"[a-zа-я0-9]+", lowered_text)
            overlap = sum(
                1
                for query_token in query_tokens
                if any(_word_matches(query_token, text_token) for text_token in text_tokens)
            )
            if "спо" in query_tokens and "средн" in lowered_text and "профессиональн" in lowered_text:
                overlap += 2
            if "конкурс" in query_tokens and ("конкурс" in lowered_text or "средн" in lowered_text):
                overlap += 2
            if "конкурс" in query_tokens and (
                "без вступительных" in lowered_text
                or "средний балл" in lowered_text
                or "документе об образовании" in lowered_text
            ):
                overlap += 4
            if "испытан" in query_tokens and "вступительн" in lowered_text:
                overlap += 1
            if overlap >= 2:
                candidates.append((overlap, _chunk_index(chunk) or 0, chunk))

        for _, _, chunk in sorted(candidates, key=lambda item: (-item[0], item[1]))[:max_extra_per_doc]:
            if len(selected) >= max_total_chunks:
                return selected, added
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            selected.append(_minimal_chunk(chunk))
            added = True

    return selected, added


def _query_tokens(question: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zа-я0-9]+", _normalize_for_match(question)):
        if token in QUESTION_STOPWORDS:
            continue
        if token.isdigit() or len(token) >= 3:
            tokens.add(token)
    return tokens


def _rank_evidence_chunks_for_question(
    question: str,
    chunks: list[dict],
    question_focus: Optional[dict] = None,
) -> list[dict]:
    question_focus = question_focus or analyze_question_focus(question)
    query_tokens = _query_tokens(question)
    focus_terms = question_focus.get("question_focus_terms", [])
    entity_type = question_focus.get("expected_entity_type", "general")
    rule = _focus_rule(entity_type)

    if not query_tokens and not focus_terms:
        return chunks

    ranked = []
    for original_rank, chunk in enumerate(chunks):
        text = _normalize_for_match(chunk.get("text", ""))
        doc_id = _normalize_for_match(chunk.get("doc_id", ""))
        text_tokens = set(re.findall(r"[a-zа-я0-9]+", text))
        doc_tokens = set(re.findall(r"[a-zа-я0-9]+", doc_id))
        overlap = len(query_tokens.intersection(text_tokens))
        doc_overlap = len(query_tokens.intersection(doc_tokens))
        score = overlap + doc_overlap * 2

        for term in focus_terms:
            if _text_contains_term(text, term):
                score += 4
            if _text_contains_term(doc_id, term):
                score += 2

        if rule:
            for marker in rule.get("positive_doc_markers", ()):
                if marker in doc_id:
                    score += 8
            for marker in rule.get("positive_text_markers", ()):
                if _text_contains_term(text, marker):
                    score += 6
            for marker in rule.get("negative_doc_markers", ()):
                if marker in doc_id:
                    score -= 7
            for marker in rule.get("negative_text_markers", ()):
                if _text_contains_term(text, marker):
                    score -= 5

        ranked.append((score, original_rank, chunk))

    return [
        chunk
        for _, _, chunk in sorted(
            ranked,
            key=lambda item: (-item[0], item[1]),
        )
    ]


def _text_contains_term(text: str, term: str) -> bool:
    lowered_text = _normalize_for_match(text)
    lowered_term = _normalize_for_match(term)
    if not lowered_term:
        return False
    if lowered_term in lowered_text:
        return True

    term_tokens = [
        token
        for token in re.findall(r"[a-zа-я0-9]+", lowered_term)
        if token not in QUESTION_STOPWORDS and (token.isdigit() or len(token) >= 4)
    ]
    if not term_tokens:
        return False

    text_tokens = re.findall(r"[a-zа-я0-9]+", lowered_text)
    matched = 0
    for term_token in term_tokens:
        if any(_word_matches(term_token, text_token) for text_token in text_tokens):
            matched += 1
    threshold = 1.0 if len(term_tokens) <= 2 else 0.65
    return matched / len(term_tokens) >= threshold


def _format_retrieved_chunks(chunks: list[dict]) -> str:
    formatted_chunks = []
    for rank, chunk in enumerate(chunks, start=1):
        formatted_chunks.append(
            "\n".join(
                [
                    f"[{rank}] chunk_id: {chunk.get('chunk_id')}",
                    f"doc_id: {chunk.get('doc_id')}",
                    "text:",
                    chunk.get("text", ""),
                ]
            )
        )
    return "\n\n---\n\n".join(formatted_chunks)


def _intent_instructions(intent: str) -> str:
    return INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["default"])


def _retry_instructions(intent: str, coverage_gate_reason: str = "") -> str:
    reason = f" Coverage gate reason: {coverage_gate_reason}" if coverage_gate_reason else ""
    if intent == "list":
        return (
            "RETRY MODE: previous fact plan was too narrow for a list question. "
            "Inspect all evidence chunks again and return one claim per relevant list item."
            f"{reason}"
        )
    if intent == "comparison":
        return (
            "RETRY MODE: previous fact plan did not cover both sides of the comparison. "
            "Return claims for both compared sides and their differences."
            f"{reason}"
        )
    if intent == "numeric":
        return (
            "RETRY MODE: previous fact plan missed numeric values. "
            "Return claims that include the relevant entity and numeric value from evidence."
            f"{reason}"
        )
    if intent == "deadline":
        return (
            "RETRY MODE: previous fact plan missed dates or times. "
            "Return claims containing the date/time and the condition it applies to."
            f"{reason}"
        )
    if intent == "condition":
        return (
            "RETRY MODE: previous fact plan missed a rule, limitation, or reason. "
            "Return separate claims for the rule and the condition/reason."
            f"{reason}"
        )
    return (
        "RETRY MODE: previous fact plan was incomplete. "
        "Return all central evidence-backed claims needed for the answer."
        f"{reason}"
    )


def _format_focus_info(question_focus: dict) -> str:
    return "\n".join(
        [
            f"question_focus_terms: {json.dumps(question_focus.get('question_focus_terms', []), ensure_ascii=False)}",
            f"expected_entity_type: {question_focus.get('expected_entity_type', 'general')}",
            f"comparison_sides: {json.dumps(question_focus.get('comparison_sides', []), ensure_ascii=False)}",
            f"numeric_targets: {json.dumps(question_focus.get('numeric_targets', []), ensure_ascii=False)}",
        ]
    )


def _build_prompt(
    question: str,
    prepared_chunks: list[dict],
    question_intent: str,
    question_focus: Optional[dict] = None,
    retry: bool = False,
    coverage_gate_reason: str = "",
) -> str:
    question_focus = question_focus or analyze_question_focus(question)
    return FACT_PLAN_PROMPT.format(
        question=question,
        retrieved_chunks=_format_retrieved_chunks(prepared_chunks),
        max_claims=FACT_PLAN_MAX_CLAIMS,
        question_intent=question_intent,
        question_focus_info=_format_focus_info(question_focus),
        intent_instructions=_intent_instructions(question_intent),
        retry_instructions=(
            _retry_instructions(question_intent, coverage_gate_reason=coverage_gate_reason)
            if retry
            else ""
        ),
    )


def _has_insufficient_evidence_note(notes: str) -> bool:
    lowered = notes.lower()
    markers = (
        "недостат",
        "нет данных",
        "нет достаточной информации",
        "не хватает evidence",
        "insufficient",
        "no evidence",
    )
    return any(marker in lowered for marker in markers)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    return bool(value)


def _normalize_for_match(text: str) -> str:
    return " ".join(str(text or "").lower().replace("ё", "е").split())


def _is_list_or_overview_question(question: str) -> bool:
    lowered = _normalize_for_match(question)
    return any(marker in lowered for marker in LIST_QUESTION_MARKERS)


def _evidence_text(chunks: list[dict]) -> str:
    return "\n".join(str(chunk.get("text", "")) for chunk in chunks)


def _evidence_has_enumeration(chunks: list[dict]) -> bool:
    text = _evidence_text(chunks)
    if LIST_INTRO_PATTERN.search(text):
        return True
    return bool(re.search(r"(^|\s)[-•]\s+\S+", text) or "|" in text)


def _evidence_has_number(chunks: list[dict]) -> bool:
    return bool(re.search(r"\d", _evidence_text(chunks)))


def _claims_have_number(fact_plan: list[dict]) -> bool:
    return any(re.search(r"\d", str(item.get("claim", ""))) for item in fact_plan)


def _claims_have_numeric_value(fact_plan: list[dict]) -> bool:
    claims_text = _fact_plan_text(fact_plan)
    return _claims_have_number(fact_plan) or any(
        marker in claims_text
        for marker in ("одно", "один", "пять", "пяти", "трех", "трёх")
    )


def _evidence_has_date_like(chunks: list[dict]) -> bool:
    return bool(DATE_LIKE_PATTERN.search(_evidence_text(chunks)))


def _claims_have_date_like(fact_plan: list[dict]) -> bool:
    return bool(DATE_LIKE_PATTERN.search(" ".join(str(item.get("claim", "")) for item in fact_plan)))


def _evidence_has_condition_marker(chunks: list[dict]) -> bool:
    lowered = _normalize_for_match(_evidence_text(chunks))
    markers = (
        "не гарант",
        "не означает",
        "потому",
        "так как",
        "при услов",
        "если",
        "только",
        "необходимо",
        "следует",
        "может отличаться",
    )
    return any(marker in lowered for marker in markers)


def _important_words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zа-я0-9]+", _normalize_for_match(text))
        if token.isdigit() or len(token) > 4
    ]


def _word_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.isdigit() or right.isdigit():
        return False
    shortest = min(len(left), len(right))
    if shortest >= 5 and left[:5] == right[:5]:
        return True
    return shortest >= 4 and left[:4] == right[:4]


def _is_duplicate_claim(claim: str, existing_claims: list[dict]) -> bool:
    claim_words = _important_words(claim)
    if not claim_words:
        return False

    for existing in existing_claims:
        existing_claim = str(existing.get("claim", ""))
        existing_words = _important_words(existing_claim)
        if not existing_words:
            continue

        matched = 0
        for claim_word in claim_words:
            if any(_word_matches(claim_word, word) for word in existing_words):
                matched += 1

        if matched / len(claim_words) >= 0.6:
            return True

    return False


def _split_list_items(items_text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", items_text)
    raw_items = re.split(r"[,;]", normalized)

    items = []
    for item in raw_items:
        cleaned = item.strip(" \t\r\n-–—:;,.")
        cleaned = re.sub(
            r"^(?:и|а также|также|следующие|следующим образом)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n-–—:;,.")
        lowered = _normalize_for_match(cleaned)
        if len(cleaned) < 4:
            continue
        if lowered.startswith(("для ", "если ", "что ", "которые ")):
            continue
        items.append(cleaned)

    deduplicated = []
    seen = set()
    for item in items:
        key = _normalize_for_match(item)
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


def _list_claim_prefix(question: str) -> str:
    lowered = _normalize_for_match(question)
    if "особенност" in lowered:
        return "К особенностям относится"
    if "бюджет" in lowered and "крупн" in lowered:
        return "К направлениям с крупным набором на бюджетные места относится"
    if "направлен" in lowered:
        return "К основным направлениям поступления относится"
    if "документ" in lowered:
        return "К релевантным документам относится"
    if "услов" in lowered:
        return "К релевантным условиям относится"
    if "вид" in lowered:
        return "К релевантным видам относится"
    return "К релевантным пунктам относится"


def _extract_feature_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "особенност" not in lowered_question:
        return []

    claims = []
    seen = set()
    order = 0
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    preferred_doc_marker = None
    if "базов" in lowered_question and "специализирован" not in lowered_question:
        preferred_doc_marker = "basic_higher"
    elif "специализирован" in lowered_question:
        preferred_doc_marker = "specialized_higher"

    feature_markers = (
        "траектор",
        "модул",
        "проект",
        "цифров",
        "практик",
        "индустр",
        "постеп",
        "первые",
        "унификац",
        "индивидуализац",
        "профессиональн",
    )

    for chunk in prepared_chunks:
        doc_id = str(chunk.get("doc_id", ""))
        if preferred_doc_marker and preferred_doc_marker not in doc_id:
            continue
        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split()).strip(" -")
            if sentence.startswith("Общая идея ") and "Базовое высшее образование" in sentence:
                sentence = sentence[sentence.index("Базовое высшее образование") :]
            if len(sentence) < 35 or len(sentence) > 430:
                continue
            lowered_sentence = _normalize_for_match(sentence)
            if not any(marker in lowered_sentence for marker in feature_markers):
                continue
            if sentence.startswith("#") or "тип документа" in lowered_sentence:
                continue
            if lowered_sentence.startswith(("он предназначен", "данный документ")):
                continue
            if sentence[:1].islower():
                continue
            key = lowered_sentence
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "claim_id": "",
                    "claim": sentence,
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено как особенность из evidence.",
                }
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims
    return claims


def _extract_comparison_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "отлич" not in lowered_question and "разниц" not in lowered_question:
        return []

    candidates = []
    seen = set()
    order = 0
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    comparison_markers = (
        "разные",
        "ориент",
        "предназнач",
        "егэ",
        "вступительн",
        "срок",
        "правил",
        "програм",
        "магистрат",
        "первый курс",
        "предыдущ",
        "уров",
    )

    for chunk in prepared_chunks:
        doc_id = str(chunk.get("doc_id", ""))
        if (
            "базов" in lowered_question
            and "специализирован" in lowered_question
            and "basic_higher" not in doc_id
            and "specialized_higher" not in doc_id
        ):
            continue

        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split()).strip(" -")
            if "Базовое высшее образование" in sentence and sentence.startswith("Отличие "):
                sentence = sentence[sentence.index("Базовое высшее образование") :]
            if sentence.startswith("Срок обучения:") and "На официальной" in sentence:
                sentence = sentence[sentence.index("На официальной") :]
            if len(sentence) < 35 or len(sentence) > 430:
                continue
            lowered_sentence = _normalize_for_match(sentence)
            if lowered_sentence.startswith(("назначение документа", "данный документ", "он предназначен")):
                continue
            if sentence[:1].islower():
                continue
            if not (
                "базов" in lowered_sentence
                or "специализирован" in lowered_sentence
                or "магистрат" in lowered_sentence
            ):
                continue
            if not any(marker in lowered_sentence for marker in comparison_markers):
                continue
            key = lowered_sentence
            if key in seen:
                continue
            seen.add(key)
            score = 0
            if "базов" in lowered_sentence and (
                "специализирован" in lowered_sentence or "магистрат" in lowered_sentence
            ):
                score += 4
            if "разные уров" in lowered_sentence or "разные траектор" in lowered_sentence:
                score += 4
            if "для базового" in lowered_sentence or "для специализирован" in lowered_sentence:
                score += 3
            if "срок" in lowered_sentence or "1 или 2" in lowered_sentence:
                score += 3
            if any(marker in lowered_sentence for marker in ("вступительн", "егэ", "правил", "програм")):
                score += 2
            if "первый курс" in lowered_sentence or "предыдущ" in lowered_sentence:
                score += 2
            if "не является" in lowered_sentence or "нельзя" in lowered_sentence:
                score += 1

            candidates.append(
                (
                    score,
                    order,
                    {
                        "claim_id": "",
                        "claim": sentence,
                        "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                        "confidence": "medium",
                        "possible_conflict": False,
                        "notes": "Выделено как отличие из comparison evidence.",
                    },
                )
            )
            order += 1

    claims = [
        item
        for _, _, item in sorted(
            candidates,
            key=lambda candidate: (-candidate[0], candidate[1]),
        )[:FACT_PLAN_MAX_CLAIMS]
    ]
    return claims


def _extract_budget_place_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "бюджет" not in lowered_question or "крупн" not in lowered_question:
        return []

    claims = []
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for match in BUDGET_PLACE_ITEM_PATTERN.finditer(text):
            name = match.group("name").strip()
            count = match.group("count").strip()
            claims.append(
                {
                    "claim_id": "",
                    "claim": (
                        f"К направлениям с крупным набором на бюджетные места "
                        f"относится «{name}», где указано {count} бюджетных мест."
                    ),
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено из явного перечисления направлений и числа бюджетных мест в evidence.",
                }
            )

        if claims:
            return claims
    return claims


def _extract_medical_program_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "медицин" not in lowered_question and "осмотр" not in lowered_question:
        return []

    claims = []
    seen = set()
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        lowered_text = _normalize_for_match(text)
        if "медицин" not in lowered_text and "осмотр" not in lowered_text:
            continue

        for match in PROGRAM_CODE_ITEM_PATTERN.finditer(text):
            code = match.group("code").strip()
            name = match.group("name").strip()
            key = f"{code} {name}"
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "claim_id": "",
                    "claim": (
                        f"Предварительный медицинский осмотр требуется при "
                        f"поступлении на {code} «{name}»."
                    ),
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено из перечня программ для предварительного медицинского осмотра в evidence.",
                }
            )

        if len(claims) >= 2:
            return claims
    return claims


def _clean_subject(value: str) -> str:
    subject = " ".join(str(value or "").split()).strip(" |:-–—.,")
    return subject[:1].lower() + subject[1:] if subject else subject


def _extract_score_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "балл" not in lowered_question:
        return []

    claims = []
    seen = set()
    seen_subjects = set()
    wants_budget = "бюджет" in lowered_question
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        lowered_text = _normalize_for_match(text)
        if "минимальн" not in lowered_text or "балл" not in lowered_text:
            continue
        if wants_budget and "платн" in lowered_text and "бюджет" not in lowered_text:
            continue

        matches = []
        for match in SCORE_TABLE_PATTERN.finditer(text):
            subject = _clean_subject(match.group("subject"))
            score = match.group("score").strip()
            if subject.lower() in {"общеобразовательный предмет", "минимальный балл"}:
                continue
            matches.append((subject, score))

        for match in SCORE_SENTENCE_PATTERN.finditer(text):
            window_start = max(0, match.start() - 120)
            window_end = min(len(text), match.end() + 80)
            window = _normalize_for_match(text[window_start:window_end])
            if wants_budget and "платн" in window and "бюджет" not in window:
                continue
            subject = _clean_subject(match.group("subject"))
            score = match.group("score").strip()
            matches.append((subject, score))

        for subject, score in matches:
            subject_key = _normalize_for_match(subject)
            if wants_budget and subject_key in seen_subjects:
                continue
            key = f"{subject}:{score}"
            if key in seen:
                continue
            seen.add(key)
            seen_subjects.add(subject_key)
            claims.append(
                {
                    "claim_id": "",
                    "claim": (
                        "Для бюджетных мест по программам базового высшего "
                        f"образования минимальный балл по {subject} составляет {score}."
                    ),
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено из числового evidence о минимальных баллах.",
                }
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims
    return claims


def _extract_money_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if not any(marker in lowered_question for marker in ("стипенд", "выплат", "рубл", "бви", "егэ")):
        return []

    claims = []
    seen = set()
    seen_amounts = set()

    def add_claim(key: str, claim: str, chunk: dict, notes: str) -> None:
        if len(claims) >= FACT_PLAN_MAX_CLAIMS:
            return
        if key in seen:
            return
        seen.add(key)
        for amount in re.findall(r"\d{1,3}(?:\s?\d{3})*", claim):
            seen_amounts.add(" ".join(amount.split()))
        claims.append(
            {
                "claim_id": "",
                "claim": claim,
                "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                "confidence": "medium",
                "possible_conflict": False,
                "notes": notes,
            }
        )

    def chunk_is_relevant(text: str) -> bool:
        lowered_text = _normalize_for_match(text)
        if "военн" in lowered_text or "вуц" in lowered_text or "целев" in lowered_text:
            return any(marker in lowered_question for marker in ("военн", "вуц", "целев"))
        return True

    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        if not chunk_is_relevant(text):
            continue
        lowered_text = _normalize_for_match(text)
        if "бви" not in lowered_text and "без вступительных испытаний" not in lowered_text:
            continue
        for match in BVI_STIPEND_PATTERN.finditer(text):
            amount = " ".join(match.group("amount").split())
            period = " в месяц" if match.group("period") else ""
            add_claim(
                key=f"bvi-stipend:{amount}:{period}",
                claim=f"Для поступающих БВИ указана повышенная стипендия {amount} рублей{period}.",
                chunk=chunk,
                notes="Выделено из evidence о повышенной стипендии для поступающих БВИ.",
            )

    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        if not chunk_is_relevant(text):
            continue
        for match in MONEY_RANGE_PATTERN.finditer(text):
            amount = " ".join(match.group("amount").split())
            condition = " ".join(match.group("condition").split()).strip(" .;")
            if "егэ" not in _normalize_for_match(condition) and "балл" not in _normalize_for_match(condition):
                continue
            add_claim(
                key=f"ege-range:{amount}:{_normalize_for_match(condition)}",
                claim=f"Выплата {amount} рублей указана для условия: {condition}.",
                chunk=chunk,
                notes="Выделено из числового evidence о выплатах по результатам ЕГЭ.",
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims

    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        if not chunk_is_relevant(text):
            continue
        for match in ONE_TIME_PAYMENT_PATTERN.finditer(text):
            amount = " ".join(match.group("amount").split())
            condition = " ".join(match.group("condition").split()).strip(" .;")
            add_claim(
                key=f"one-time:{amount}:{_normalize_for_match(condition)}",
                claim=(
                    f"Единоразовая выплата после зачисления в размере {amount} рублей "
                    f"указана для абитуриентов, {condition}."
                ),
                chunk=chunk,
                notes="Выделено из evidence о единоразовой выплате после зачисления.",
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims

    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        if not chunk_is_relevant(text):
            continue
        for match in MONEY_SENTENCE_PATTERN.finditer(text):
            sentence = " ".join(match.group(0).split()).strip()
            lowered_sentence = _normalize_for_match(sentence)
            if not any(marker in lowered_sentence for marker in ("бви", "егэ", "стипенд", "выплат")):
                continue
            sentence_amounts = {
                " ".join(amount.split())
                for amount in re.findall(r"\d{1,3}(?:\s?\d{3})*", sentence)
            }
            if sentence_amounts and sentence_amounts.issubset(seen_amounts):
                continue
            add_claim(
                key=f"sentence:{lowered_sentence}",
                claim=sentence,
                chunk=chunk,
                notes="Выделено из предложения evidence о стипендиях или выплатах.",
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims
    return claims


def _extract_numeric_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    claims.extend(_extract_score_claims(question, prepared_chunks))
    if len(claims) >= FACT_PLAN_MAX_CLAIMS:
        return claims[:FACT_PLAN_MAX_CLAIMS]

    seen_budget_claims = set()
    for item in _extract_budget_place_claims(question, prepared_chunks):
        if len(claims) >= FACT_PLAN_MAX_CLAIMS:
            break
        claim_key = _normalize_for_match(item["claim"])
        if claim_key not in seen_budget_claims:
            seen_budget_claims.add(claim_key)
            claims.append(item)

    for item in _extract_money_claims(question, prepared_chunks):
        if len(claims) >= FACT_PLAN_MAX_CLAIMS:
            break
        claims.append(item)
    return claims


def _sentence_claims_by_markers(
    prepared_chunks: list[dict],
    markers: tuple[str, ...],
    max_claims: int = 3,
) -> list[dict]:
    claims = []
    seen = set()
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split())
            lowered = _normalize_for_match(sentence)
            if not any(marker in lowered for marker in markers):
                continue
            if len(sentence) < 30 or len(sentence) > 360:
                continue
            key = _normalize_for_match(sentence)
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "claim_id": "",
                    "claim": sentence,
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено из evidence по маркерам intent.",
                }
            )
            if len(claims) >= max_claims:
                return claims
    return claims


def _extract_condition_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    markers = (
        "не гарант",
        "не означает",
        "ориентир",
        "прошлого года",
        "может отличаться",
        "только",
        "при услов",
        "если",
        "необходимо",
    )
    if not any(marker in lowered_question for marker in CONDITION_QUESTION_MARKERS) and not any(
        marker in lowered_question for marker in ("гарант", "почему")
    ):
        return []
    return _sentence_claims_by_markers(prepared_chunks, markers, max_claims=4)


def _extract_deadline_claims(question: str, prepared_chunks: list[dict]) -> list[dict]:
    if classify_question_intent(question) != "deadline":
        return []
    claims = []
    for item in _sentence_claims_by_markers(
        prepared_chunks,
        ("срок", "прием", "приём", "документ", "конкурс", "зачислен"),
        max_claims=FACT_PLAN_MAX_CLAIMS,
    ):
        if DATE_LIKE_PATTERN.search(item["claim"]):
            claims.append(item)
    return claims


def _extract_list_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
) -> list[dict]:
    if not _is_list_or_overview_question(question):
        return []

    feature_claims = _extract_feature_claims_from_evidence(question, prepared_chunks)
    if feature_claims:
        return feature_claims

    medical_claims = _extract_medical_program_claims(question, prepared_chunks)
    if medical_claims:
        return medical_claims

    budget_claims = _extract_budget_place_claims(question, prepared_chunks)
    if budget_claims:
        return budget_claims

    prefix = _list_claim_prefix(question)
    claims = []
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for match in LIST_INTRO_PATTERN.finditer(text):
            items_text = match.group("items")
            stripped_items = items_text.strip()
            if stripped_items.endswith(("—", "-")):
                continue
            if stripped_items.count("«") != stripped_items.count("»"):
                continue

            items = _split_list_items(items_text)
            if len(items) < 2:
                continue

            for item in items:
                claim = f"{prefix} {item}."
                claims.append(
                    {
                        "claim_id": "",
                        "claim": claim,
                        "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                        "confidence": "medium",
                        "possible_conflict": False,
                        "notes": "Выделено из явного перечисления в evidence.",
                    }
                )
            if claims:
                return claims
    return claims


def _extract_overview_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if not any(
        marker in lowered_question
        for marker in ("роль", "раздел", "как правильно", "как проходит", "проходит", "порядок")
    ):
        return []

    if "официальн" in lowered_question and "документ" in lowered_question:
        prefix = "Раздел официальных документов включает"
    elif "переч" in lowered_question and "испытан" in lowered_question:
        prefix = "При чтении перечня вступительных испытаний нужно учитывать"
    elif "конкурс" in lowered_question:
        prefix = "В процедуре конкурса учитывается"
    else:
        prefix = "К важным пунктам относится"

    claims = []
    seen = set()
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for match in LIST_INTRO_PATTERN.finditer(text):
            items_text = match.group("items")
            if ";" in items_text:
                items = [
                    re.sub(
                        r"^(?:и|а также|также)\s+",
                        "",
                        item.strip(" \t\r\n-–—:;,."),
                        flags=re.IGNORECASE,
                    )
                    for item in items_text.split(";")
                ]
                items = [item for item in items if len(item) >= 4]
            else:
                items = _split_list_items(items_text)
            if len(items) < 2:
                continue
            for item in items:
                claim = f"{prefix} {item}."
                key = _normalize_for_match(claim)
                if key in seen:
                    continue
                seen.add(key)
                claims.append(
                    {
                        "claim_id": "",
                        "claim": claim,
                        "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                        "confidence": "medium",
                        "possible_conflict": False,
                        "notes": "Выделено из явного перечисления overview evidence.",
                    }
                )
                if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                    return claims
            if claims:
                return claims
    return claims


def _extract_exam_reading_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "переч" not in lowered_question or "испытан" not in lowered_question:
        return []
    if "читать" not in lowered_question and "правильно" not in lowered_question:
        return []

    claims = []
    seen = set()
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    markers = (
        "предметы нужны",
        "обязательные предметы",
        "предметы по выбору",
        "косую черту",
        "скобках",
        "математика",
        "русский язык",
        "профильные предметы",
        "название программы",
        "кодом",
        "уровнем образования",
    )

    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split()).strip(" -")
            if "Перечень вступительных испытаний" in sentence and sentence[:1].islower():
                sentence = sentence[sentence.index("Перечень вступительных испытаний") :]
            if len(sentence) < 35 or len(sentence) > 430:
                continue
            lowered_sentence = _normalize_for_match(sentence)
            if not any(marker in lowered_sentence for marker in markers):
                continue
            if sentence[:1].islower():
                continue
            key = lowered_sentence
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "claim_id": "",
                    "claim": sentence,
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено как правило чтения перечня вступительных испытаний.",
                }
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims
    return claims


def _extract_spo_competition_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
) -> list[dict]:
    lowered_question = _normalize_for_match(question)
    if "спо" not in lowered_question and "средн" not in lowered_question:
        return []
    if "конкурс" not in lowered_question and "проходит" not in lowered_question:
        return []

    claims = []
    seen = set()
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    strong_markers = (
        "без вступительных испытаний",
        "средний балл",
        "среднего балла",
        "документе об образовании",
        "документа об образовании",
        "поступающих больше",
        "превышает количество бюджетных мест",
        "конкурсной процедуре",
    )

    for chunk in prepared_chunks:
        if "secondary_vocational" not in str(chunk.get("doc_id", "")):
            continue
        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split()).strip(" -")
            if "Поступление на СПО" in sentence and sentence[:1].islower():
                sentence = sentence[sentence.index("Поступление на СПО") :]
            if "Зачисление проводится" in sentence and sentence[:1].islower():
                sentence = sentence[sentence.index("Зачисление проводится") :]
            if len(sentence) < 35 or len(sentence) > 430:
                continue
            lowered_sentence = _normalize_for_match(sentence)
            if not any(marker in lowered_sentence for marker in strong_markers):
                continue
            if any(marker in lowered_sentence for marker in ("ошибка", "адрес", "срок")) and not any(
                marker in lowered_sentence
                for marker in ("без вступительных", "средний балл", "документе об образовании")
            ):
                continue
            if sentence[:1].islower():
                continue
            key = lowered_sentence
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "claim_id": "",
                    "claim": sentence,
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено как правило конкурса на программы СПО.",
                }
            )
            if len(claims) >= FACT_PLAN_MAX_CLAIMS:
                return claims
    return claims


def _add_focus_claim(
    claims: list[dict],
    seen: set[str],
    claim: str,
    chunk: dict,
    notes: str,
) -> None:
    if len(claims) >= FACT_PLAN_MAX_CLAIMS:
        return
    cleaned = " ".join(str(claim or "").split()).strip()
    if not cleaned:
        return
    key = _normalize_for_match(cleaned)
    if key in seen:
        return
    seen.add(key)
    claims.append(
        {
            "claim_id": "",
            "claim": cleaned,
            "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
            "confidence": "medium",
            "possible_conflict": False,
            "notes": notes,
        }
    )


def _focus_sentences(
    prepared_chunks: list[dict],
    markers: tuple[str, ...],
    max_claims: int,
    excluded_markers: tuple[str, ...] = (),
) -> list[dict]:
    claims = []
    seen = set()
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split()).strip(" -")
            lowered = _normalize_for_match(sentence)
            if not any(_text_contains_term(lowered, marker) for marker in markers):
                continue
            if any(_text_contains_term(lowered, marker) for marker in excluded_markers):
                continue
            if len(sentence) < 25 or len(sentence) > 430:
                continue
            _add_focus_claim(
                claims,
                seen,
                sentence,
                chunk,
                "Выделено focus-aware extractor из prepared evidence.",
            )
            if len(claims) >= max_claims:
                return claims
    return claims


def _extract_application_limit_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "одно заявление" not in lowered and "пять направ" not in lowered:
            continue
        if "одно заявление" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "В МАИ поступающий может подать только одно заявление о приёме.",
                chunk,
                "Выделено как ограничение по числу заявлений.",
            )
        if "не более пяти" in lowered and ("направлен" in lowered or "специальност" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "В одном заявлении можно указать не более пяти направлений подготовки или специальностей.",
                chunk,
                "Выделено как ограничение по числу направлений.",
            )
        if len(claims) >= 2:
            return claims
    return claims


def _extract_target_obligation_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "целевое обучение осуществляется на основании договора" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Целевое обучение осуществляется на основании договора с заказчиком.",
                chunk,
                "Выделено как основание целевого обучения.",
            )
        if "обязан освоить образовательную программу" in lowered and "отработать не менее" in lowered:
            _add_focus_claim(
                claims,
                seen,
                (
                    "Студент обязан освоить образовательную программу и отработать "
                    "не менее трёх лет на предприятии, указанном в договоре."
                ),
                chunk,
                "Выделено как обязательство студента по договору целевого обучения.",
            )
        if len(claims) >= 2:
            return claims
    return claims


def _extract_score_comparison_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "минимальн" not in lowered and "проходн" not in lowered:
            continue
        if "минимальный балл" in lowered and ("порог" in lowered or "прохождение вступительного" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "Минимальный балл — это нижний порог успешного прохождения вступительного испытания по отдельному предмету.",
                chunk,
                "Выделено как определение минимального балла.",
            )
        if "проходной балл" in lowered and "итог" in lowered and "конкурс" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Проходной балл формируется по итогам конкретного конкурса и заранее точно неизвестен.",
                chunk,
                "Выделено как определение проходного балла.",
            )
        if (
            "сам по себе не означает" in lowered
            or "порог допуска" in lowered
            or "не гарант" in lowered
        ) and "минимальн" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Минимальный балл является порогом допуска и сам по себе не гарантирует зачисление.",
                chunk,
                "Выделено как ограничение минимального балла.",
            )
        if len(claims) >= 3:
            return claims
    return claims


def _extract_enrollment_comparison_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "конкурсн" not in lowered and "приказ" not in lowered:
            continue
        if "конкурсные списки" in lowered and "сумм" in lowered and "балл" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Конкурсный список показывает поступающих, участвующих в конкурсе, и их положение по сумме конкурсных баллов.",
                chunk,
                "Выделено как назначение конкурсного списка.",
            )
        if "не является приказом" in lowered or "не является фактом зачисления" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Конкурсный список не является фактом зачисления.",
                chunk,
                "Выделено как отличие конкурсного списка от приказа.",
            )
        if "факт зачисления" in lowered and ("приказ" in lowered or "издания приказа" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "Факт зачисления подтверждается только приказом о зачислении.",
                chunk,
                "Выделено как роль приказа о зачислении.",
            )
        if len(claims) >= 3:
            return claims
    return claims


def _extract_quota_comparison_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "квот" not in lowered:
            continue
        if "особая квота" in lowered and ("социальн" in lowered or "инвалид" in lowered):
            _add_focus_claim(
                claims,
                seen,
                (
                    "Особая квота предназначена для отдельных социальных и правовых "
                    "категорий поступающих, например инвалидов, детей-сирот и отдельных ветеранов боевых действий."
                ),
                chunk,
                "Выделено как описание особой квоты.",
            )
        if "отдельная квота" in lowered and ("5.1" in lowered or "273" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "Отдельная квота связана с категориями лиц, указанными в части 5.1 статьи 71 Федерального закона № 273-ФЗ.",
                chunk,
                "Выделено как описание отдельной квоты.",
            )
        if "целевая квота" in lowered and "договор" in lowered and "заказчик" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Целевая квота связана с целевым обучением, заказчиком и договором о целевом обучении.",
                chunk,
                "Выделено как описание целевой квоты.",
            )
        if len(claims) >= 3:
            return claims
    return claims


def _extract_paid_budget_plan_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "план при" not in lowered and "бюджетн" not in lowered and "платн" not in lowered:
            continue
        if "бюджет" in lowered and ("федерального бюджета" in lowered or "федерального бюджет" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "План бюджетных мест показывает места, финансируемые за счёт федерального бюджета.",
                chunk,
                "Выделено как назначение плана бюджетных мест.",
            )
        if "платн" in lowered and (
            "средств физических" in lowered
            or "юридических лиц" in lowered
            or "договорам об образовании" in lowered
            or "договору об образовании" in lowered
        ):
            _add_focus_claim(
                claims,
                seen,
                "План платных мест показывает места по договорам об образовании за счёт средств физических или юридических лиц.",
                chunk,
                "Выделено как назначение плана платных мест.",
            )
        if "количество бюджетных и платных мест" in lowered and "отличаться" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Количество бюджетных и платных мест по одному направлению может различаться.",
                chunk,
                "Выделено как различие количества мест.",
            )
        if len(claims) >= 3:
            return claims
    return claims


def _extract_network_program_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "сетев" not in lowered:
            continue
        if "участие нескольких организаций" in lowered or "нескольких организаций" in lowered:
            _add_focus_claim(
                claims,
                seen,
                (
                    "В отличие от обычной образовательной программы, сетевая программа "
                    "реализуется с участием нескольких организаций, например базового вуза, "
                    "вуза-партнёра и предприятия."
                ),
                chunk,
                "Выделено как базовое отличие сетевой программы.",
            )
        if "практик" in lowered and ("партнер" in lowered or "партн" in lowered or "производственной баз" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "В сетевой программе часть обучения или практики может проходить у партнёров или на производственной базе.",
                chunk,
                "Выделено как отличие организации обучения и практики.",
            )
        if "не отменяет конкурс" in lowered or ("вступительные испытания" in lowered and "минимальных баллов" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "Сетевая программа не отменяет конкурс, вступительные испытания, минимальные баллы и сроки.",
                chunk,
                "Выделено как ограничение сетевой программы при поступлении.",
            )
        if len(claims) >= 3:
            return claims
    return claims


def _extract_career_opportunity_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "карьер" not in lowered and "крыл" not in lowered and "работодател" not in lowered:
            continue
        if "карьерные возможности" in lowered and (
            "работодател" in lowered or "индустриальн" in lowered
        ):
            _add_focus_claim(
                claims,
                seen,
                "На сайте представлены карьерные возможности, связанные с работодателями и индустриальными партнёрами.",
                chunk,
                "Выделено как обзор карьерных возможностей.",
            )
        if "день карьеры" in lowered and "один день" in lowered and "профориентацион" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Среди форматов упоминаются День карьеры, материалы «Один день в…» и профориентационные мероприятия предприятий.",
                chunk,
                "Выделено как перечень карьерных форматов.",
            )
        if "крылья ростеха" in lowered and ("авиастроительн" in lowered or "инженер" in lowered):
            _add_focus_claim(
                claims,
                seen,
                "Проект «Крылья Ростеха» связан с подготовкой инженеров для отечественной авиастроительной отрасли.",
                chunk,
                "Выделено как индустриальный проект.",
            )
        if len(claims) >= 3:
            return claims
    return claims


def _extract_admission_direction_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "базовое высшее" in lowered and "специализированное высшее" in lowered and "среднее профессиональное" in lowered:
            _add_focus_claim(
                claims,
                seen,
                (
                    "На сайте приёмной комиссии МАИ выделены базовое высшее образование, "
                    "специализированное высшее образование, среднее профессиональное образование "
                    "и раздел для иностранных граждан."
                ),
                chunk,
                "Выделено как перечень основных направлений поступления.",
            )
        if "точн" in lowered and "официальн" in lowered and "специализирован" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Для точных сведений о сроках, экзаменах, документах и количестве мест используются специализированные разделы и официальные документы.",
                chunk,
                "Выделено как правило использования разделов сайта.",
            )
        if len(claims) >= 2:
            return claims
    return claims


def _extract_applicant_document_claims(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    seen = set()
    for chunk in prepared_chunks:
        lowered = _normalize_for_match(chunk.get("text", ""))
        if "поступающий представляет" not in lowered and "при подаче заявления" not in lowered:
            continue
        if "личность" in lowered and "гражданство" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Поступающий представляет документ, удостоверяющий личность и гражданство.",
                chunk,
                "Выделено как обязательный документ поступающего.",
            )
        if "документ установленного образца" in lowered and "образован" in lowered:
            _add_focus_claim(
                claims,
                seen,
                "Поступающий представляет документ установленного образца об образовании.",
                chunk,
                "Выделено как обязательный документ об образовании.",
            )
        if len(claims) >= 2:
            return claims
    return claims


def _extract_focus_aware_claims_from_evidence(
    question: str,
    prepared_chunks: list[dict],
    question_focus: dict,
) -> list[dict]:
    entity_type = question_focus.get("expected_entity_type", "general")
    if entity_type == "admission_directions":
        return _extract_admission_direction_claims(prepared_chunks)
    if entity_type == "applicant_documents":
        return _extract_applicant_document_claims(prepared_chunks)
    if entity_type == "application_limits":
        return _extract_application_limit_claims(prepared_chunks)
    if entity_type == "target_obligations":
        return _extract_target_obligation_claims(prepared_chunks)
    if entity_type == "scores" and question_focus.get("comparison_sides"):
        return _extract_score_comparison_claims(prepared_chunks)
    if entity_type == "enrollment_list_order":
        return _extract_enrollment_comparison_claims(prepared_chunks)
    if entity_type == "quotas":
        return _extract_quota_comparison_claims(prepared_chunks)
    if entity_type == "paid_vs_budget_plan":
        return _extract_paid_budget_plan_claims(prepared_chunks)
    if entity_type == "network_programs":
        return _extract_network_program_claims(prepared_chunks)
    if entity_type == "career_opportunities":
        return _extract_career_opportunity_claims(prepared_chunks)
    return []


def _extract_guidance_claims_from_evidence(prepared_chunks: list[dict]) -> list[dict]:
    claims = []
    sentence_pattern = re.compile(r"[^.!?]*[.!?]", flags=re.DOTALL)
    for chunk in prepared_chunks:
        text = str(chunk.get("text", ""))
        for sentence_match in sentence_pattern.finditer(text):
            sentence = " ".join(sentence_match.group(0).split())
            lowered = _normalize_for_match(sentence)
            if "точн" not in lowered:
                continue
            if "официальн" not in lowered or "специализирован" not in lowered:
                continue
            if len(sentence) > 320:
                continue

            claims.append(
                {
                    "claim_id": "",
                    "claim": sentence,
                    "evidence_chunk_ids": [str(chunk.get("chunk_id"))],
                    "confidence": "medium",
                    "possible_conflict": False,
                    "notes": "Выделено как правило использования источников из evidence.",
                }
            )
            return claims
    return claims


def _renumber_fact_plan(fact_plan: list[dict]) -> list[dict]:
    for index, item in enumerate(fact_plan, start=1):
        item["claim_id"] = f"P{index:03d}"
    return fact_plan


def _augment_fact_plan_for_coverage(
    question: str,
    fact_plan: list[dict],
    prepared_chunks: list[dict],
    question_intent: str,
    question_focus: Optional[dict] = None,
) -> list[dict]:
    question_focus = question_focus or analyze_question_focus(question)
    augmented = []
    seen_list_claims = set()

    heuristic_claims = []
    focus_aware_claims = _extract_focus_aware_claims_from_evidence(
        question,
        prepared_chunks,
        question_focus,
    )
    if focus_aware_claims:
        focused_plan = _renumber_fact_plan(focus_aware_claims[:FACT_PLAN_MAX_CLAIMS])
        gate = _coverage_gate_result(
            question,
            question_focus,
            focused_plan,
            prepared_chunks,
        )
        if gate["passed"]:
            return focused_plan
        heuristic_claims.extend(focus_aware_claims)

    if question_intent == "numeric":
        heuristic_claims.extend(_extract_numeric_claims(question, prepared_chunks))
    elif question_intent == "deadline":
        heuristic_claims.extend(_extract_deadline_claims(question, prepared_chunks))
    elif question_intent == "condition":
        heuristic_claims.extend(_extract_condition_claims(question, prepared_chunks))

    if question_intent == "list" or (question_intent == "numeric" and not heuristic_claims):
        heuristic_claims.extend(_extract_list_claims_from_evidence(question, prepared_chunks))
    elif question_intent == "default" and len(fact_plan) < 2:
        heuristic_claims.extend(_extract_exam_reading_claims_from_evidence(question, prepared_chunks))
        heuristic_claims.extend(_extract_spo_competition_claims_from_evidence(question, prepared_chunks))
        heuristic_claims.extend(_extract_overview_claims_from_evidence(question, prepared_chunks))

    if question_intent == "comparison":
        comparison_claims = _extract_comparison_claims_from_evidence(
            question,
            prepared_chunks,
        )
        if comparison_claims:
            heuristic_claims.extend(comparison_claims)
        elif len(fact_plan) < 2:
            heuristic_claims.extend(
                _sentence_claims_by_markers(
                    prepared_chunks,
                    ("базов", "специализирован", "магистрат", "отлич"),
                    max_claims=4,
                )
            )

    for item in heuristic_claims:
        if len(augmented) >= FACT_PLAN_MAX_CLAIMS:
            break
        claim_key = _normalize_for_match(item["claim"])
        if claim_key not in seen_list_claims:
            seen_list_claims.add(claim_key)
            augmented.append(item)

    for item in _extract_guidance_claims_from_evidence(prepared_chunks):
        if len(augmented) >= FACT_PLAN_MAX_CLAIMS:
            break
        if not _is_duplicate_claim(item["claim"], augmented):
            augmented.append(item)

    for item in fact_plan:
        if len(augmented) >= FACT_PLAN_MAX_CLAIMS:
            break
        if not _is_duplicate_claim(item["claim"], augmented):
            augmented.append(item)

    if not augmented:
        return _renumber_fact_plan(fact_plan)

    return _renumber_fact_plan(augmented)


def _normalize_evidence_ids(
    evidence_ids: list[str],
    valid_chunk_ids: set[str],
    doc_to_chunk_ids: dict,
) -> list[str]:
    normalized_ids = []
    for evidence_id in evidence_ids:
        if evidence_id in valid_chunk_ids:
            normalized_ids.append(evidence_id)
            continue

        for chunk_id in doc_to_chunk_ids.get(evidence_id, []):
            normalized_ids.append(chunk_id)

    deduplicated_ids = []
    for evidence_id in normalized_ids:
        if evidence_id not in deduplicated_ids:
            deduplicated_ids.append(evidence_id)
    return deduplicated_ids


def _normalize_fact_plan(
    parsed,
    prepared_chunks: Optional[list[dict]] = None,
) -> tuple[list[dict], Optional[str]]:
    if isinstance(parsed, dict):
        for key in ("fact_plan", "claims", "plan"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
        else:
            if "claim" in parsed:
                parsed = [parsed]

    if not isinstance(parsed, list):
        return [], "Parsed JSON is not a list."

    prepared_chunks = prepared_chunks or []
    valid_chunk_ids = {
        str(chunk.get("chunk_id"))
        for chunk in prepared_chunks
        if chunk.get("chunk_id")
    }
    doc_to_chunk_ids = {}
    for chunk in prepared_chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        if doc_id and chunk_id:
            doc_to_chunk_ids.setdefault(str(doc_id), []).append(str(chunk_id))

    fact_plan = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        notes = str(item.get("notes", "")).strip()
        evidence_ids = item.get("evidence_chunk_ids", [])

        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        evidence_ids = [str(chunk_id).strip() for chunk_id in evidence_ids if chunk_id]
        if prepared_chunks:
            evidence_ids = _normalize_evidence_ids(
                evidence_ids,
                valid_chunk_ids,
                doc_to_chunk_ids,
            )

        if not claim:
            continue
        if not evidence_ids:
            continue

        confidence = str(item.get("confidence", "low")).strip().lower()
        if confidence not in ALLOWED_CONFIDENCE:
            confidence = "low"

        fact_plan.append(
            {
                "claim_id": str(item.get("claim_id") or f"P{index:03d}").strip(),
                "claim": claim,
                "evidence_chunk_ids": evidence_ids,
                "confidence": confidence,
                "possible_conflict": _to_bool(item.get("possible_conflict", False)),
                "notes": notes,
            }
        )

    return fact_plan[:FACT_PLAN_MAX_CLAIMS], None


def _find_marker_chunk(chunks_by_doc: dict, markers: tuple[str, ...]):
    for doc_id, chunks in chunks_by_doc.items():
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            if any(marker in text for marker in markers):
                return doc_id, chunk
    return None, None


def _maybe_mark_guarantee_conflict(
    question: str,
    fact_plan: list[dict],
    prepared_chunks: list[dict],
) -> None:
    lowered_question = question.lower()
    if not any(marker in lowered_question for marker in ("гарант", "автомат", "всем", "право")):
        return

    chunks_by_doc = {}
    for chunk in prepared_chunks:
        chunks_by_doc.setdefault(chunk.get("doc_id"), []).append(chunk)

    negative_doc_id, negative_chunk = _find_marker_chunk(
        chunks_by_doc,
        NEGATIVE_GUARANTEE_MARKERS,
    )
    positive_doc_id, positive_chunk = _find_marker_chunk(
        {
            doc_id: chunks
            for doc_id, chunks in chunks_by_doc.items()
            if doc_id != negative_doc_id
        },
        POSITIVE_PROVISION_MARKERS,
    )

    if not negative_chunk or not positive_chunk:
        return

    conflict_note = (
        "Prepared evidence содержит расхождение по вопросу гарантии или "
        "автоматического предоставления: один источник говорит об отсутствии "
        "гарантии/конкурсной основе, другой формулирует предоставление места "
        "для соответствующей категории студентов."
    )

    for item in fact_plan:
        item["possible_conflict"] = True
        evidence_ids = item["evidence_chunk_ids"]
        for chunk in (negative_chunk, positive_chunk):
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in evidence_ids:
                evidence_ids.append(chunk_id)
        if conflict_note not in item["notes"]:
            item["notes"] = (item["notes"] + " " + conflict_note).strip()


def _comparison_sides_covered(question: str, fact_plan: list[dict]) -> bool:
    lowered_question = _normalize_for_match(question)
    claims_text = _normalize_for_match(" ".join(str(item.get("claim", "")) for item in fact_plan))

    required_markers = []
    if "базов" in lowered_question:
        required_markers.append("базов")
    if "специализирован" in lowered_question:
        required_markers.append("специализирован")
    if "минимальн" in lowered_question:
        required_markers.append("минимальн")
    if "проходн" in lowered_question:
        required_markers.append("проходн")
    if "бюджет" in lowered_question:
        required_markers.append("бюджет")
    if "платн" in lowered_question:
        required_markers.append("платн")

    if len(required_markers) >= 2:
        return all(marker in claims_text for marker in required_markers)
    return len(fact_plan) >= 2


def _fact_plan_text(fact_plan: list[dict]) -> str:
    return _normalize_for_match(" ".join(str(item.get("claim", "")) for item in fact_plan))


def _claim_is_relevant_to_focus(claim: str, question_focus: dict) -> bool:
    entity_type = question_focus.get("expected_entity_type", "general")
    lowered = _normalize_for_match(claim)

    if entity_type == "admission_directions":
        return any(
            marker in lowered
            for marker in ("базов", "специализирован", "средн", "профессиональн", "иностран")
        )
    if entity_type == "applicant_documents":
        if any(marker in lowered for marker in ("нормативн", "план прием", "план приём", "вступительн")):
            return False
        return "документ" in lowered and any(
            marker in lowered
            for marker in ("личност", "гражданств", "образован", "снилс", "индивидуальн")
        )
    if entity_type == "target_obligations":
        return any(
            marker in lowered
            for marker in ("договор", "заказчик", "обязан", "освоить", "отработать")
        )
    if entity_type == "career_opportunities":
        return any(
            marker in lowered
            for marker in ("карьер", "работодател", "день карьеры", "один день", "крылья", "индустриальн")
        )
    if entity_type == "quotas":
        return "квот" in lowered
    return any(
        _text_contains_term(lowered, term)
        for term in question_focus.get("question_focus_terms", [])
    )


def _relevant_claim_count(fact_plan: list[dict], question_focus: dict) -> int:
    return sum(
        1
        for item in fact_plan
        if _claim_is_relevant_to_focus(str(item.get("claim", "")), question_focus)
    )


def _evidence_has_focus_marker(prepared_chunks: list[dict], marker: str) -> bool:
    return any(_text_contains_term(str(chunk.get("text", "")), marker) for chunk in prepared_chunks)


def _coverage_gate_result(
    question: str,
    question_focus: dict,
    fact_plan: list[dict],
    prepared_chunks: list[dict],
) -> dict:
    if not fact_plan:
        return {"passed": False, "reason": "fact_plan is empty"}

    intent = question_focus.get("intent") or classify_question_intent(question)
    entity_type = question_focus.get("expected_entity_type", "general")
    claims_text = _fact_plan_text(fact_plan)
    lowered_question = _normalize_for_match(question)

    if entity_type == "admission_directions":
        required = ("базов", "специализирован", "средн", "иностран")
        missing = [marker for marker in required if marker not in claims_text]
        if missing:
            return {
                "passed": False,
                "reason": f"admission direction plan misses markers: {', '.join(missing)}",
            }

    if entity_type == "applicant_documents":
        if "образован" not in claims_text or "личност" not in claims_text:
            return {
                "passed": False,
                "reason": "applicant document plan must include identity/citizenship and education document",
            }

    if entity_type == "application_limits":
        missing = []
        if "заявлен" in lowered_question and not (
            "одно заявлен" in claims_text or "только одно заявлен" in claims_text
        ):
            missing.append("one application")
        if "направлен" in lowered_question and not (
            ("пять" in claims_text or "пяти" in claims_text or "5" in claims_text)
            and "направлен" in claims_text
        ):
            missing.append("five directions")
        if missing:
            return {"passed": False, "reason": f"numeric targets missing: {', '.join(missing)}"}

    if entity_type == "target_obligations":
        missing = []
        if "договор" not in claims_text or "заказчик" not in claims_text:
            missing.append("contract/customer")
        if "освоить" not in claims_text and "образовательн" not in claims_text:
            missing.append("complete educational program")
        if "отработ" not in claims_text:
            missing.append("work obligation")
        if missing:
            return {"passed": False, "reason": f"target obligations missing: {', '.join(missing)}"}

    if entity_type == "scores" and question_focus.get("comparison_sides"):
        missing = [
            side
            for side in question_focus["comparison_sides"]
            if not _text_contains_term(claims_text, side)
        ]
        if missing:
            return {"passed": False, "reason": f"comparison sides missing: {', '.join(missing)}"}
        if "порог" not in claims_text or "конкурс" not in claims_text:
            return {"passed": False, "reason": "score comparison lacks threshold/competition distinction"}

    if entity_type == "enrollment_list_order":
        missing = []
        if "конкурсн" not in claims_text or "спис" not in claims_text:
            missing.append("competition list")
        if "приказ" not in claims_text:
            missing.append("enrollment order")
        if "факт зачислен" not in claims_text and "не является" not in claims_text:
            missing.append("enrollment fact distinction")
        if missing:
            return {"passed": False, "reason": f"enrollment comparison missing: {', '.join(missing)}"}
        if "ошиб" in claims_text and "конкурсн" not in claims_text:
            return {"passed": False, "reason": "plan drifted to wrong-order examples"}

    if entity_type == "quotas":
        missing = []
        if "особ" in lowered_question and not any(marker in claims_text for marker in ("социаль", "инвалид", "сирот")):
            missing.append("special quota categories")
        if "отдельн" in lowered_question and not any(marker in claims_text for marker in ("5.1", "273", "статьи 71")):
            missing.append("separate quota legal basis")
        if "целев" in lowered_question and not ("договор" in claims_text and "заказчик" in claims_text):
            missing.append("target quota contract/customer")
        if missing:
            return {"passed": False, "reason": f"quota comparison missing: {', '.join(missing)}"}

    if entity_type == "paid_vs_budget_plan":
        missing = []
        if "федеральн" not in claims_text or "бюджет" not in claims_text:
            missing.append("budget funding")
        if "платн" not in claims_text or "договор" not in claims_text:
            missing.append("paid contract places")
        if "количество" not in claims_text or (
            "отлич" not in claims_text and "различ" not in claims_text
        ):
            missing.append("different place counts")
        if "минимальн" in claims_text and "балл" in claims_text:
            missing.append("plan drifted to minimum scores")
        if missing:
            return {"passed": False, "reason": f"paid/budget plan missing: {', '.join(missing)}"}

    if entity_type == "network_programs":
        missing = []
        if "нескольк" not in claims_text and "организац" not in claims_text:
            missing.append("multiple organizations")
        if "практик" not in claims_text and "производствен" not in claims_text:
            missing.append("practice/production base")
        if "не отменяет" not in claims_text and "конкурс" not in claims_text:
            missing.append("admission rules limitation")
        if missing:
            return {"passed": False, "reason": f"network comparison missing: {', '.join(missing)}"}

    if entity_type == "career_opportunities":
        missing = []
        if "работодател" not in claims_text and "индустриальн" not in claims_text:
            missing.append("employers/industrial partners")
        if _evidence_has_focus_marker(prepared_chunks, "крылья ростеха") and "крыл" not in claims_text:
            missing.append("Krylya Rostekha project")
        if missing:
            return {"passed": False, "reason": f"career overview missing: {', '.join(missing)}"}

    if intent == "list":
        relevant_count = _relevant_claim_count(fact_plan, question_focus)
        if _evidence_has_enumeration(prepared_chunks) and relevant_count < 2:
            return {
                "passed": False,
                "reason": "list question has enumeration in evidence but fewer than 2 relevant plan items",
            }

    if intent == "comparison":
        sides = question_focus.get("comparison_sides", [])
        if sides:
            missing_sides = [
                side for side in sides if not _text_contains_term(claims_text, side)
            ]
            if missing_sides:
                return {
                    "passed": False,
                    "reason": f"comparison sides missing: {', '.join(missing_sides)}",
                }
        elif not _comparison_sides_covered(question, fact_plan):
            return {"passed": False, "reason": "comparison sides are not covered"}

    if intent == "numeric":
        targets = question_focus.get("numeric_targets", [])
        missing_targets = [
            target
            for target in targets
            if not _text_contains_term(claims_text, target)
        ]
        if missing_targets:
            return {
                "passed": False,
                "reason": f"numeric targets missing: {', '.join(missing_targets)}",
            }
        if targets and not _claims_have_numeric_value(fact_plan) and _evidence_has_number(prepared_chunks):
            return {"passed": False, "reason": "numeric question has evidence numbers but plan has no numbers"}

    if intent == "deadline":
        if not _claims_have_date_like(fact_plan) and _evidence_has_date_like(prepared_chunks):
            return {"passed": False, "reason": "deadline evidence has dates but plan has no dates"}

    if intent == "condition":
        if len(fact_plan) < 2 and _evidence_has_condition_marker(prepared_chunks):
            return {"passed": False, "reason": "condition question misses rule or limitation/reason"}

    return {"passed": True, "reason": "coverage gate passed"}


def _needs_fact_plan_retry(
    question: str,
    question_intent: str,
    fact_plan: list[dict],
    prepared_chunks: list[dict],
    question_focus: Optional[dict] = None,
) -> bool:
    question_focus = question_focus or analyze_question_focus(question)
    gate = _coverage_gate_result(question, question_focus, fact_plan, prepared_chunks)
    if not gate["passed"]:
        return True

    if question_intent == "list":
        return len(fact_plan) < 2 and _evidence_has_enumeration(prepared_chunks)

    if question_intent == "comparison":
        return not _comparison_sides_covered(question, fact_plan)

    if question_intent == "numeric":
        return not _claims_have_numeric_value(fact_plan) and _evidence_has_number(prepared_chunks)

    if question_intent == "deadline":
        return not _claims_have_date_like(fact_plan) and _evidence_has_date_like(prepared_chunks)

    if question_intent == "condition":
        return len(fact_plan) < 2 and _evidence_has_condition_marker(prepared_chunks)

    return False


def _call_fact_planner_once(
    question: str,
    prepared_chunks: list[dict],
    question_intent: str,
    question_focus: Optional[dict] = None,
    llm_client=None,
    retry: bool = False,
    coverage_gate_reason: str = "",
) -> dict:
    question_focus = question_focus or analyze_question_focus(question)
    prompt_chunk_limit = 12 if retry else 10
    prompt_chunks = prepared_chunks[:prompt_chunk_limit]
    prompt = _build_prompt(
        question=question,
        prepared_chunks=prompt_chunks,
        question_intent=question_intent,
        question_focus=question_focus,
        retry=retry,
        coverage_gate_reason=coverage_gate_reason,
    )

    result = {
        "fact_plan": [],
        "raw_output": "",
        "error": None,
        "latency_sec": 0.0,
        "num_llm_calls": 1,
    }

    def use_heuristic_plan(raw_output: str = "") -> bool:
        heuristic_plan = _augment_fact_plan_for_coverage(
            question,
            [],
            prepared_chunks,
            question_intent,
            question_focus=question_focus,
        )
        if not heuristic_plan:
            return False
        _maybe_mark_guarantee_conflict(question, heuristic_plan, prepared_chunks)
        result["fact_plan"] = heuristic_plan
        result["raw_output"] = raw_output
        result["error"] = None
        return True

    if use_heuristic_plan():
        gate = _coverage_gate_result(
            question,
            question_focus,
            result["fact_plan"],
            prepared_chunks,
        )
        if gate["passed"]:
            result["num_llm_calls"] = 0
            return result

    try:
        llm_result = _call_llm(prompt, llm_client=llm_client)
    except Exception as exc:
        if use_heuristic_plan():
            return result
        result["error"] = f"Fact planning LLM call failed: {exc}"
        return result

    raw_output = llm_result.get("text", "")
    result["raw_output"] = raw_output
    result["latency_sec"] = llm_result.get("latency_sec", 0.0)

    if llm_result.get("error"):
        if use_heuristic_plan(raw_output=raw_output):
            return result
        result["error"] = llm_result["error"]
        return result

    parsed, parse_error = safe_json_loads(raw_output, fallback=[])
    if parse_error:
        if use_heuristic_plan(raw_output=raw_output):
            return result
        result["error"] = parse_error
        return result

    fact_plan, normalize_error = _normalize_fact_plan(parsed, prepared_chunks)
    fact_plan = _augment_fact_plan_for_coverage(
        question,
        fact_plan,
        prepared_chunks,
        question_intent,
        question_focus=question_focus,
    )
    _maybe_mark_guarantee_conflict(question, fact_plan, prepared_chunks)

    result["fact_plan"] = fact_plan
    result["error"] = normalize_error
    return result


def build_fact_plan(
    question: str,
    retrieved_chunks: list[dict],
    mode: Optional[str] = None,
    llm_client=None,
    max_chunks_per_doc: int = 4,
) -> dict:
    question_focus = analyze_question_focus(question)
    question_intent = question_focus["intent"]
    evidence_mode = _infer_mode(retrieved_chunks, mode=mode)
    base_chunks = prepare_evidence_chunks(
        retrieved_chunks,
        max_chunks_per_doc=max_chunks_per_doc,
    )
    prepared_chunks, neighbor_expansion_used = expand_neighbor_chunks(
        base_chunks,
        mode=evidence_mode,
    )
    prepared_chunks, query_expansion_used = expand_query_matched_same_doc_chunks(
        prepared_chunks,
        mode=evidence_mode,
        question=question,
    )
    prepared_chunks = _rank_evidence_chunks_for_question(
        question,
        prepared_chunks,
        question_focus=question_focus,
    )
    result = {
        "fact_plan": [],
        "prepared_evidence_chunks": prepared_chunks,
        "prepared_evidence_chunk_ids": [
            str(chunk.get("chunk_id"))
            for chunk in prepared_chunks
            if chunk.get("chunk_id")
        ],
        "raw_output": "",
        "error": None,
        "latency_sec": 0.0,
        "num_llm_calls": 1,
        "question_intent": question_intent,
        "question_focus_terms": question_focus.get("question_focus_terms", []),
        "comparison_sides": question_focus.get("comparison_sides", []),
        "numeric_targets": question_focus.get("numeric_targets", []),
        "expected_entity_type": question_focus.get("expected_entity_type", "general"),
        "fact_plan_retry_used": False,
        "coverage_gate_passed": False,
        "coverage_gate_reason": "coverage gate not evaluated",
        "prepared_evidence_count": len(prepared_chunks),
        "neighbor_expansion_used": neighbor_expansion_used or query_expansion_used,
    }

    first_result = _call_fact_planner_once(
        question,
        prepared_chunks,
        question_intent,
        question_focus=question_focus,
        llm_client=llm_client,
        retry=False,
    )
    result["raw_output"] = first_result.get("raw_output", "")
    result["latency_sec"] = first_result.get("latency_sec", 0.0)
    result["num_llm_calls"] = first_result.get("num_llm_calls", 1)
    result["fact_plan"] = first_result.get("fact_plan", [])
    result["error"] = first_result.get("error")

    if result["error"]:
        return result

    gate = _coverage_gate_result(
        question,
        question_focus,
        result["fact_plan"],
        prepared_chunks,
    )
    result["coverage_gate_passed"] = bool(gate["passed"])
    result["coverage_gate_reason"] = str(gate["reason"])

    if not gate["passed"] or _needs_fact_plan_retry(
        question,
        question_intent,
        result["fact_plan"],
        prepared_chunks,
        question_focus=question_focus,
    ):
        retry_base_chunks = prepare_evidence_chunks(
            retrieved_chunks,
            max_chunks_per_doc=max(max_chunks_per_doc, 6),
        )
        retry_chunks, retry_neighbor_used = expand_neighbor_chunks(
            retry_base_chunks,
            mode=evidence_mode,
            max_total_chunks=24,
        )
        retry_chunks, retry_query_used = expand_query_matched_same_doc_chunks(
            retry_chunks,
            mode=evidence_mode,
            question=question,
            max_total_chunks=28,
        )
        retry_chunks = _rank_evidence_chunks_for_question(
            question,
            retry_chunks,
            question_focus=question_focus,
        )
        retry_result = _call_fact_planner_once(
            question,
            retry_chunks,
            question_intent,
            question_focus=question_focus,
            llm_client=llm_client,
            retry=True,
            coverage_gate_reason=result["coverage_gate_reason"],
        )
        result["num_llm_calls"] += retry_result.get("num_llm_calls", 0)
        result["latency_sec"] += retry_result.get("latency_sec", 0.0)
        result["fact_plan_retry_used"] = True
        result["prepared_evidence_chunks"] = retry_chunks
        result["prepared_evidence_chunk_ids"] = [
            str(chunk.get("chunk_id"))
            for chunk in retry_chunks
            if chunk.get("chunk_id")
        ]
        result["prepared_evidence_count"] = len(retry_chunks)
        result["neighbor_expansion_used"] = (
            neighbor_expansion_used
            or query_expansion_used
            or retry_neighbor_used
            or retry_query_used
        )
        if retry_result.get("raw_output"):
            result["raw_output"] = retry_result["raw_output"]
        if retry_result.get("error"):
            result["error"] = retry_result["error"]
            return result
        result["fact_plan"] = retry_result.get("fact_plan", [])
        gate = _coverage_gate_result(
            question,
            question_focus,
            result["fact_plan"],
            retry_chunks,
        )
        result["coverage_gate_passed"] = bool(gate["passed"])
        result["coverage_gate_reason"] = str(gate["reason"])

    return result
