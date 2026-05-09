# Evidence-First RAG Verification

Репозиторий содержит исходный код выпускной квалификационной работы на тему:

**«Разработка и экспериментальная оценка прототипа системы Retrieval-Augmented Generation (RAG) с фактологической верификацией»**

Основной программный результат проекта — Python-библиотека RAG-системы с фактологической верификацией. Библиотека реализует финальный пайплайн **Final Evidence-First Verified RAG**, включающий поиск по локальной базе знаний, построение fact plan, генерацию ответа, claim-level verification и selective correction.

Микросервис на FastAPI является демонстрационной HTTP-оболочкой над библиотекой и не рассматривается как production-ready backend-сервис.

---

## 1. Назначение проекта

Обычные RAG-системы позволяют подключать внешнюю базу знаний к языковой модели, однако наличие найденного контекста не гарантирует фактологическую корректность итогового ответа. Модель может использовать источник неполно, неверно обобщать условия, добавлять неподдержанные утверждения или выбирать одну из конфликтующих версий без указания неопределённости.

В этом проекте реализован evidence-first подход:

1. сначала выполняется retrieval по локальному корпусу;
2. затем строится fact plan из evidence-backed claims;
3. ответ генерируется на основе этого плана;
4. итоговые claims проверяются по evidence;
5. неподдержанные, частично поддержанные или конфликтные утверждения корректируются через selective correction.

---

## 2. Что реализовано

В проекте реализованы:

- базовый RAG-пайплайн;
- улучшенный RAG-пайплайн с hybrid retrieval;
- post-hoc verification пайплайн;
- финальный Final Evidence-First Verified RAG;
- BM25 retrieval;
- vector retrieval;
- hybrid retrieval через Reciprocal Rank Fusion;
- построение fact plan;
- генерация ответа на основе fact plan;
- claim extraction;
- claim-level verification;
- selective correction;
- расчёт экспериментальных метрик;
- Python-библиотека с фасадом `VerifiedRAGSystem`;
- демонстрационная FastAPI-оболочка.

---

## 3. Структура репозитория

```text
.
├── data/
│   ├── raw/                 # основной корпус документов
│   ├── raw_conflict/        # дополнительные conflict-документы
│   └── eval/                # вопросы и оценочные данные
├── docs/                    # дополнительные описания и материалы проекта
├── results/
│   ├── main/                # результаты эксперимента на основной базе знаний
│   ├── conflict/            # результаты conflict-эксперимента
│   └── metrics_summary.csv  # сводные метрики
├── scripts/                 # сценарии подготовки данных, индексации, проверки и экспериментов
├── src/
│   ├── api/                 # демонстрационная FastAPI-оболочка
│   ├── data/                # загрузка и подготовка документов
│   ├── evaluation/          # расчёт метрик
│   ├── indexing/            # построение индекса
│   ├── llm/                 # взаимодействие с Ollama
│   ├── pipelines/           # экспериментальные RAG-пайплайны
│   ├── planning/            # evidence-first planning
│   ├── retrieval/           # vector, BM25 и hybrid retrieval
│   ├── verification/        # claim extraction, verification и correction
│   ├── library.py           # публичный фасад библиотеки
│   └── schemas.py           # dataclass-схемы библиотеки
├── requirements.txt
└── README.md
````

---

## 4. Установка

Проект рассчитан на локальный запуск в виртуальном окружении Python.

Локальное окружение, использованное при разработке:

```text
Python 3.11.15
```

Создание и активация окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установка зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

---

## 5. Внешние зависимости

Для запуска полного пайплайна требуется локально запущенная Ollama и доступная модель:

```text
qwen3.5:9b
```

Проверка доступности Ollama:

```bash
python3 scripts/00_test_ollama.py
```

---

## 6. Подготовка данных

Подготовка данных выполняется отдельным сценарием:

```bash
python3 scripts/01_prepare_data.py
```

Сценарий формирует обработанные документы и chunks для двух режимов:

* `clean` — основной корпус;
* `conflict` — основной корпус + дополнительные документы с потенциально противоречивыми сведениями.

---

## 7. Построение индекса

Построение clean-индекса:

```bash
python3 scripts/02_build_index.py --mode clean
```

Построение conflict-индекса:

```bash
python3 scripts/02_build_index.py --mode conflict
```

Если embedding-модель отсутствует в локальном кэше, можно разрешить загрузку:

```bash
python3 scripts/02_build_index.py --mode clean --allow-download
python3 scripts/02_build_index.py --mode conflict --allow-download
```

По умолчанию индекс Chroma считается генерируемым артефактом и не хранится в репозитории.

---

## 8. Проверка retrieval

Пример проверки hybrid retrieval в clean-режиме:

```bash
python3 scripts/03_test_retrieval.py \
  --mode clean \
  --retriever hybrid \
  --query "Какие минимальные баллы по информатике и физике?"
```

Пример проверки hybrid retrieval в conflict-режиме:

```bash
python3 scripts/03_test_retrieval.py \
  --mode conflict \
  --retriever hybrid \
  --query "Гарантируется ли общежитие всем иногородним студентам?" \
  --top-k 10
```

---

## 9. Использование библиотеки

Основная точка входа в библиотеку — класс `VerifiedRAGSystem`.

Пример использования:

```python
from src.library import VerifiedRAGSystem
from src.schemas import RAGRequest

rag = VerifiedRAGSystem(mode="clean")

request = RAGRequest(
    question="Когда начинается приём документов в МАИ?",
    mode="clean",
    top_k=5,
)

response = rag.ask(request)

print(response.final_answer or response.answer)
print(response.verification_report)
print(response.errors)
```

Smoke-test библиотечного API:

```bash
python3 scripts/07_test_library_api.py
```

---

## 10. Запуск экспериментов

Основной эксперимент:

```bash
python3 scripts/04_run_main_eval.py --pipeline baseline
python3 scripts/04_run_main_eval.py --pipeline advanced
python3 scripts/04_run_main_eval.py --pipeline posthoc
python3 scripts/04_run_main_eval.py --pipeline final_verified
```

Conflict-эксперимент:

```bash
python3 scripts/05_run_conflict_eval.py --pipeline baseline
python3 scripts/05_run_conflict_eval.py --pipeline advanced
python3 scripts/05_run_conflict_eval.py --pipeline posthoc
python3 scripts/05_run_conflict_eval.py --pipeline final_verified
```

Для быстрой проверки можно использовать ограниченный запуск:

```bash
python3 scripts/04_run_main_eval.py --pipeline final_verified --limit 2
python3 scripts/05_run_conflict_eval.py --pipeline final_verified --limit 1
```

Сбор метрик:

```bash
python3 scripts/06_build_metrics.py
```

В результате формируются:

```text
results/metrics_summary.csv
results/examples_for_thesis.md
```

---

## 11. FastAPI-оболочка

Микросервисная оболочка реализована как демонстрационный HTTP-интерфейс над библиотекой.

Запуск API:

```bash
python3 scripts/08_run_api.py
```

Альтернативный запуск:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Swagger UI доступен по адресу:

```text
http://127.0.0.1:8000/docs
```

Проверка API:

```bash
python3 scripts/09_test_api.py
```

Реализованные endpoint’ы:

| Endpoint  | Метод | Назначение                                                          |
| --------- | ----- | ------------------------------------------------------------------- |
| `/health` | GET   | базовая проверка состояния сервиса                                  |
| `/ask`    | POST  | запуск `VerifiedRAGSystem.ask()`                                    |
| `/verify` | POST  | проверка готового ответа по переданным evidence                     |
| `/index`  | POST  | демонстрационная заглушка; HTTP-перестроение индекса не выполняется |

---

## 12. Ограничения

Проект является исследовательским прототипом и не является production-ready сервисом.

Ограничения:

* не реализованы авторизация и пользовательские аккаунты;
* не реализован отдельный web UI;
* не реализован production monitoring;
* не реализовано масштабирование;
* HTTP endpoint `/index` не перестраивает индекс, а возвращает сообщение о необходимости выполнить индексацию отдельно;
* Chroma index должен быть подготовлен заранее;
* Ollama должна быть запущена отдельно;
* модель `qwen3.5:9b` должна быть доступна локально;
* embedding-модель должна быть доступна локально или загружена при построении индекса;
* проект не оформлен как устанавливаемый Python package через `pyproject.toml`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.