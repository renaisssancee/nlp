# CineMatch — Рекомендательная система фильмов на основе RAG

CineMatch - это система рекомендаций фильмов, которая помогает подобрать кино под ваше настроение или запрос. Пользователь может просто описать, что ему хочется посмотреть (на русском или английском), а система подберёт подходящие варианты.
В основе лежит подход Retrieval-Augmented Generation (RAG): сначала система ищет релевантные фильмы в датасете TMDB 5000, а затем формирует понятное объяснение, почему именно эти фильмы подходят под запрос.

---

## Оглавление

1. [Архитектура](#1-архитектура)
2. [Стек технологий](#2-стек-технологий)
3. [Структура проекта](#3-структура-проекта)
4. [Установка и запуск](#4-установка-и-запуск)
5. [Конфигурация](#5-конфигурация)
6. [Компоненты системы](#6-компоненты-системы)
   - [6.1 Query Analyzer](#61-query-analyzer-srcquery_analyzerpy)
   - [6.2 Retriever](#62-retriever-srcretrievalpy)
   - [6.3 Hallucination Guard](#63-hallucination-guard-srchallucinationpy)
   - [6.4 RAG Orchestrator](#64-rag-orchestrator-srcragpy)
   - [6.5 LLM Utils](#65-llm-utils-srcllm_utilspy)
   - [6.6 Streamlit UI](#66-streamlit-ui-srcapppy)
7. [Скрипты](#7-скрипты)
   - [7.1 Ingest](#71-ingest-scriptsingestpy)
   - [7.2 Build Index](#72-build-index-scriptsbuild_indexpy)
   - [7.3 Evaluate](#73-evaluate-scriptsevaluatepy)
8. [Данные](#8-данные)
9. [Промпты](#9-промпты)
10. [Обработка ошибок и отказоустойчивость](#10-обработка-ошибок-и-отказоустойчивость)
11. [Логирование и обратная связь](#11-логирование-и-обратная-связь)
12. [Оценка качества](#12-оценка-качества)
13. [Примеры работы пайплайна](#13-примеры-работы-пайплайна)

---

## 1. Архитектура

Система реализует пятиступенчатый RAG-пайплайн:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Пользователь                            │
│              "Хочу страшный фильм до 100 минут"                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. Query Analyzer (LLM)                                        │
│     Парсит запрос -> структурированные параметры                  │
│     {genre: "Horror", max_duration: 100,                        │
│      semantic_query: "scary horror movie"}                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. Hybrid Retrieval                                            │
│     Векторный поиск (ChromaDB) + Фильтры по метаданным          │
│     -> 20 кандидатов                                             │
│                                                                  │
│  3. Cross-Encoder Reranking                                     │
│     Переранжирование кандидатов -> top-5                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. Hallucination Guard                                         │
│     Проверка качества выдачи (similarity ≥ 0.4)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. Generation (LLM)                                            │
│     Генерация ответа с объяснениями                              │
│     на основе найденных фильмов                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Ответ пользователю + Логирование в SQLite                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Стек технологий

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| Векторная БД | ChromaDB ≥ 0.4.22 | Хранение и поиск эмбеддингов фильмов |
| Эмбеддинги | sentence-transformers ≥ 2.3.0 (`all-MiniLM-L6-v2`) | Векторизация текста, 384-мерные векторы |
| Реранкер | sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Точное ранжирование кандидатов |
| LLM API | openai ≥ 1.0.0 (через OpenRouter) | Анализ запросов и генерация ответов |
| Web UI | Streamlit ≥ 1.30.0 | Интерактивный чат-интерфейс |
| Данные | pandas ≥ 2.1.0, kagglehub ≥ 0.2.0 | Загрузка и обработка TMDB 5000 |
| Конфигурация | PyYAML ≥ 6.0, python-dotenv ≥ 1.0.0 | Настройки и секреты |
| Логирование | SQLite (встроенный) | Запись запросов, ответов, фидбека |
| Python | 3.11+ | Рантайм |

**LLM-модели (через OpenRouter, free tier):**

| Приоритет | Модель | Роль |
|-----------|--------|------|
| Primary | `nvidia/nemotron-3-super-120b-a12b:free` | Основная модель |
| Fallback 1 | `deepseek/deepseek-chat-v3-0324:free` | Первая резервная |
| Fallback 2 | `google/gemma-3-27b-it:free` | Вторая резервная |
| Fallback 3 | `meta-llama/llama-4-maverick:free` | Третья резервная |

---

## 3. Структура проекта

```
rag_film/
├── .env.example                 # Шаблон переменных окружения
├── .gitignore                   # Правила исключения из Git
├── config.yaml                  # Конфигурация (модели, параметры, пути)
├── prompts.yaml                 # Шаблоны промптов для LLM (v1.0)
├── requirements.txt             # Python-зависимости
├── design_doc.md                # Дизайн-документ проекта
│
├── src/                         # Основной пакет приложения
│   ├── __init__.py              # Маркер пакета
│   ├── app.py                   # Streamlit UI — точка входа для пользователя
│   ├── query_analyzer.py        # Анализ запроса -> структурированные параметры
│   ├── retrieval.py             # Векторный поиск + реранкинг
│   ├── rag.py                   # Оркестратор RAG-пайплайна
│   ├── hallucination.py         # Защита от галлюцинаций
│   └── llm_utils.py             # Общий хелпер для LLM-вызовов с retry/fallback
│
├── scripts/                     # Утилитарные скрипты
│   ├── ingest.py                # Загрузка и предобработка TMDB 5000
│   ├── build_index.py           # Построение ChromaDB-индекса
│   └── evaluate.py              # Оценка качества пайплайна
│
├── data/
│   ├── raw/                     # Исходные CSV от TMDB (заполняется ingest.py)
│   ├── processed/
│   │   └── movies.jsonl         # Обработанные фильмы (~5000 записей, ~5.3 МБ)
│   ├── test_queries.jsonl       # Тестовый набор (30 запросов с разметкой)
│   └── logs.db                  # SQLite-лог запросов (создаётся автоматически)
│
└── chroma_db/                   # Персистентный векторный индекс (создаётся build_index.py)
```

---

## 4. Установка и запуск

### Предварительные требования

- Python 3.11+
- API-ключ OpenRouter ([openrouter.ai](https://openrouter.ai))

### Шаг 1. Клонирование и установка зависимостей

```bash
git clone <repo-url>
cd rag_film
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Шаг 2. Настройка API-ключа

```bash
cp .env.example .env
# Отредактируйте .env и вставьте свой ключ:
# OPENROUTER_API_KEY=sk-or-v1-...
```

### Шаг 3. Загрузка данных и построение индекса

```bash
# Скачивает TMDB 5000 с Kaggle и создаёт movies.jsonl
python scripts/ingest.py

# Строит векторный индекс в ChromaDB
python scripts/build_index.py
```

### Шаг 4. Запуск приложения

```bash
streamlit run src/app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`.

### Шаг 5. (Опционально) Запуск оценки качества

```bash
python scripts/evaluate.py
```

---

## 5. Конфигурация

Вся конфигурация сосредоточена в двух YAML-файлах.

### config.yaml — параметры системы

```yaml
# Модели
embedding_model: "all-MiniLM-L6-v2"          # Модель эмбеддингов (384 измерения)
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Кросс-энкодер для реранкинга
llm_model: "nvidia/nemotron-3-super-120b-a12b:free"      # Основная LLM
fallback_models:                               # Резервные LLM (в порядке приоритета)
  - "deepseek/deepseek-chat-v3-0324:free"
  - "google/gemma-3-27b-it:free"
  - "meta-llama/llama-4-maverick:free"
openrouter_base_url: "https://openrouter.ai/api/v1"

# ChromaDB
chroma_db_path: "chroma_db"                   # Путь к персистентной БД
chroma_collection: "movies"                    # Имя коллекции
chroma_distance: "cosine"                      # Метрика расстояния

# Поиск
retrieval:
  n_results: 20                # Начальное кол-во кандидатов из ChromaDB
  top_k: 5                    # Финальное кол-во рекомендаций
  min_results_with_filter: 5  # Мин. результатов с фильтром (иначе -> поиск без фильтра)
  similarity_threshold: 0.4   # Порог сходства (ниже -> "ничего не найдено")

# Анализ запроса
query_analyzer:
  max_retries: 2               # Макс. повторов при ошибке парсинга JSON
  backoff_base_seconds: 2      # База экспоненциального backoff (2^1=2с, 2^2=4с, ...)

# Генерация ответа
generation:
  max_retries: 2               # Макс. повторов при ошибке парсинга JSON
  backoff_base_seconds: 2      # База экспоненциального backoff
  history_turns: 2             # Кол-во пар (вопрос-ответ) для контекста диалога

# Данные
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  movies_file: "data/processed/movies.jsonl"

# Логирование
logging:
  db_path: "data/logs.db"
```

**Ключевые настройки для тюнинга:**

| Параметр | Влияние | Компромисс |
|----------|---------|------------|
| `similarity_threshold` | Порог для hallucination guard | Ниже -> больше ответов, но выше риск нерелевантных рекомендаций |
| `n_results` | Размер пула кандидатов | Больше -> точнее реранкинг, но медленнее |
| `top_k` | Количество рекомендаций | Больше -> больше выбор, но менее фокусированный ответ |
| `history_turns` | Глубина контекста диалога | Больше -> лучше понимание контекста, но длиннее промпт |

### prompts.yaml — шаблоны промптов

Содержит system-промпты и user-шаблоны для двух LLM-вызовов: анализа запроса и генерации ответа. Подробнее в разделе [9. Промпты](#9-промпты).

---

## 6. Компоненты системы

### 6.1 Query Analyzer (`src/query_analyzer.py`)

**Назначение:** Преобразование свободного текстового запроса пользователя в структурированные параметры поиска с помощью LLM.

**Класс: `QueryAnalyzer`**

```python
class QueryAnalyzer:
    def __init__(self, api_key: str)
    def analyze(self, user_query: str, history: list[dict] | None = None) -> dict
    def _clean_json_response(self, text: str) -> str
```

**Метод `analyze()` — основной метод:**

1. Формирует контекст диалога из последних `history_turns * 2` сообщений.
2. Подставляет запрос и историю в шаблон промпта.
3. Вызывает LLM через `llm_call_with_retry()` (с fallback-моделями и exponential backoff).
4. Парсит JSON-ответ, очищая от markdown code fences.
5. При ошибке парсинга — повторяет до `max_retries` раз.
6. При полном отказе — возвращает запрос "как есть" в поле `semantic_query`.

**Формат выходных данных:**

```python
# Успешный парсинг — рекомендация:
{
    "genre": "Horror",           # жанр или None
    "mood": "scary",             # настроение или None
    "max_duration": 100,         # макс. длительность (мин.) или None
    "min_year": 2010,            # мин. год выпуска или None
    "min_rating": 7.0,           # мин. рейтинг или None
    "semantic_query": "scary horror movie"  # всегда на английском
}

# Off-topic запрос:
{"off_topic": True}

# Ошибка парсинга (fallback):
{
    "genre": None, "mood": None, "max_duration": None,
    "min_year": None, "min_rating": None,
    "semantic_query": "исходный запрос пользователя"
}
```

**Метод `_clean_json_response()`:**

Очищает ответ LLM от markdown-обёрток:
- Удаляет ` ```json ` и ` ``` `
- Извлекает первый JSON-объект `{...}` с помощью регулярного выражения
- Возвращает очищенный текст для `json.loads()`

---

### 6.2 Retriever (`src/retrieval.py`)

**Назначение:** Гибридный поиск фильмов: векторное сходство + фильтрация по метаданным + кросс-энкодерный реранкинг.

**Класс: `Retriever`**

```python
class Retriever:
    def __init__(self)
    def retrieve(self, parsed_query: dict) -> list[dict]
    def _build_where_filter(self, parsed_query: dict) -> dict | None
    def _query_chroma(self, query_embedding: list, where_filter: dict | None) -> list[dict]
    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]
```

**Инициализация:**
- Загружает SentenceTransformer (`all-MiniLM-L6-v2`) для создания эмбеддингов.
- Загружает CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) для реранкинга.
- Подключается к персистентному ChromaDB и получает коллекцию `movies`.

**Метод `retrieve()` — основной пайплайн:**

```
semantic_query -> Encode -> ChromaDB query (+ metadata filters)
                              │
                        20 кандидатов
                              │
                    [если < 5 -> повтор без фильтра]
                              │
                     Cross-Encoder Reranking
                              │
                        top-5 результатов
```

1. Кодирует `semantic_query` в вектор
2. Строит фильтр метаданных (жанр, длительность, год, рейтинг)
3. Запрашивает ChromaDB, получает до 20 кандидатов
4. Если с фильтром найдено менее 5, повторяет без фильтра
5. Реранжирует кандидатов кросс-энкодером
6. Возвращает top-5

**Метод `_build_where_filter()` — построение ChromaDB-фильтров:**

Поддерживаемые фильтры:

| Поле | Оператор ChromaDB | Пример |
|------|-------------------|--------|
| `genre` | `$contains` по `genres_pipe` | `|Horror|` содержит "Horror" |
| `max_duration` | `$lte` по `duration_min` | `duration_min ≤ 100` |
| `min_year` | `$gte` по `year` | `year ≥ 2010` |
| `min_rating` | `$gte` по `rating` | `rating ≥ 7.0` |

При нескольких условиях объединяются через `$and`.

**Метод `_rerank()` — кросс-энкодерное переранжирование:**

- Формирует пары `(запрос, описание_фильма)` для каждого кандидата.
- CrossEncoder оценивает семантическую релевантность каждой пары.
- Сортирует по `rerank_score` (убывание).

**Формат выходного объекта фильма:**

```python
{
    "id": "19995",
    "title": "Avatar",
    "year": 2009,
    "duration_min": 162,
    "rating": 7.2,
    "genres": "|Action|Adventure|Fantasy|Science Fiction|",
    "overview": "In the 22nd century, a paraplegic Marine...",
    "similarity": 0.7234,    # косинусное сходство (из ChromaDB)
    "rerank_score": 8.45     # оценка кросс-энкодера
}
```

---

### 6.3 Hallucination Guard (`src/hallucination.py`)

**Назначение:** Предотвращение нерелевантных рекомендаций. Если лучший найденный фильм слишком далёк от запроса, система честно сообщает, что подходящих результатов нет.

**Функция:**

```python
def check_retrieval_quality(candidates: list[dict]) -> tuple[bool, str]
```

**Логика:**
1. Если список кандидатов пуст -> `(False, "подходящих фильмов не найдено")`
2. Находит максимальное значение `similarity` среди всех кандидатов.
3. Если `max_similarity < 0.4` -> `(False, "попробуйте переформулировать")`
4. Иначе -> `(True, "")` — качество достаточное.

**Зачем это нужно:** Без этой проверки LLM может "натянуть" объяснение на нерелевантные фильмы, создавая иллюзию полезного ответа. Hallucination guard гарантирует, что LLM получает только достаточно релевантных кандидатов.

---

### 6.4 RAG Orchestrator (`src/rag.py`)

**Назначение:** Центральный компонент, объединяющий все этапы пайплайна. Управляет потоком данных от запроса до ответа, логированием и обратной связью.

**Класс: `CineMatchRAG`**

```python
class CineMatchRAG:
    def __init__(self, api_key: str)
    def query(self, user_query: str, history: list[dict] | None = None) -> dict
    def save_feedback(self, request_id: str, feedback: str)
```

**Метод `query()` — главная точка входа:**

```python
def query(self, user_query, history=None) -> dict:
    # 1. Генерация request_id (UUID) и замер времени
    # 2. Анализ запроса (QueryAnalyzer)
    #    -> если off_topic — ранний возврат
    # 3. Поиск кандидатов (Retriever)
    # 4. Проверка качества (Hallucination Guard)
    #    -> если low quality — возврат с fallback-сообщением
    # 5. Генерация ответа (LLM)
    # 6. Логирование в SQLite
    # 7. Возврат результата
```

**Формат ответа:**

```python
# Рекомендация:
{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "type": "recommendation",
    "message": "Вот несколько фильмов, которые могут вам понравиться:",
    "movies": [
        {
            "title": "Insidious",
            "year": 2010,
            "rating": 6.8,
            "duration_min": 103,
            "reason": "Классический хоррор с психологическими пугалками"
        },
        ...
    ]
}

# Off-topic:
{
    "request_id": "...",
    "type": "off_topic",
    "message": "Я — CineMatch, рекомендательная система фильмов...",
    "movies": []
}

# Нет результатов:
{
    "request_id": "...",
    "type": "no_results",
    "message": "По вашему запросу подходящих фильмов не найдено...",
    "movies": []
}
```

**Метод `_generate_response()` — генерация ответа:**

1. Форматирует найденные фильмы в JSON.
2. Подставляет в шаблон промпта: запрос, фильмы, историю.
3. Вызывает LLM через `llm_call_with_retry()`.
4. Парсит JSON-ответ.
5. **Fallback**: если LLM не отвечает или JSON невалиден — возвращает базовую информацию о фильмах (без генеративных объяснений).

**Внутренние методы:**
- `_init_db()` — создание SQLite-таблицы `logs` при инициализации.
- `_log(...)` — запись запроса/ответа в SQLite.
- `save_feedback(request_id, feedback)` — сохранение лайка/дизлайка от пользователя.
- `_clean_json_response(text)` — очистка ответа LLM от markdown.

---

### 6.5 LLM Utils (`src/llm_utils.py`)

**Назначение:** Общий хелпер для надёжных LLM-вызовов с exponential backoff и цепочкой fallback-моделей. Используется в `query_analyzer.py`, `rag.py` и `evaluate.py`.

**Функция:**

```python
def llm_call_with_retry(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    fallback_models: list[str] | None = None,
    max_retries: int = 2,
    backoff_base: float = 2.0,
) -> str | None
```

**Алгоритм:**

```
Для каждой модели в [primary, fallback_1, fallback_2, ...]:
    Для каждой попытки (0 .. max_retries):
        Попытка вызова API
        ├── Успех -> return текст ответа
        ├── RateLimitError / APIConnectionError / APIStatusError
        │   ├── Есть ещё попытки -> sleep(backoff_base^(attempt+1)), retry
        │   └── Попытки кончились -> следующая модель
        └── Пустой ответ -> следующая модель

Все модели исчерпаны -> return None
```

**Задержки при backoff (backoff_base=2):**
- 1-й retry: `2^1 = 2` секунды
- 2-й retry: `2^2 = 4` секунды
- 3-й retry: `2^3 = 8` секунд

**Обрабатываемые ошибки:**
- `openai.RateLimitError` (HTTP 429) — превышение лимита запросов
- `openai.APIConnectionError` — проблемы с сетью
- `openai.APIStatusError` — серверные ошибки API (500, 503 и др.)
- `ValueError` — пустой ответ от модели

---

### 6.6 Streamlit UI (`src/app.py`)

**Назначение:** Веб-интерфейс с чатом для взаимодействия с пользователем.

**Основные элементы:**

- **Заголовок:** "CineMatch" с подписью "Рекомендательная система фильмов на основе RAG"
- **Чат:** Многоходовый диалог с сохранением истории в `st.session_state`
- **Ввод:** Текстовое поле с плейсхолдером "Опишите, какой фильм вы хотите посмотреть..."
- **Рекомендации:** Форматированный вывод с рейтингом, длительностью и объяснением
- **Обратная связь:** Кнопки "👍" и "👎" на каждом ответе ассистента

**Формат отображения рекомендации:**

```
**1. Insidious** (2010)
   ⭐ 6.8 | ⏱ 103 мин
   _Классический хоррор с психологическими пугалками_
```

**Обработка ошибок:**

Вызов `rag.query()` обёрнут в `try/except`. При любом необработанном исключении пользователь видит дружелюбное сообщение вместо traceback:

> "Произошла ошибка при обработке запроса. Попробуйте ещё раз через несколько секунд."

**Кэширование:**

`get_rag()` декорирован `@st.cache_resource` — RAG-пайплайн (включая загрузку моделей и подключение к ChromaDB) инициализируется один раз и переиспользуется между запросами.

---

## 7. Скрипты

### 7.1 Ingest (`scripts/ingest.py`)

**Назначение:** Загрузка датасета TMDB 5000 с Kaggle и его предобработка.

**Запуск:** `python scripts/ingest.py`

**Что делает:**

1. Скачивает `tmdb_5000_movies.csv` через `kagglehub`.
2. Парсит JSON-столбцы (`genres`, `keywords`), извлекая имена.
3. Извлекает год из `release_date`.
4. Переименовывает: `runtime` -> `duration_min`, `vote_average` -> `rating`.
5. Фильтрует фильмы без описания или длительности.
6. Создаёт `text_for_embedding`:
   ```
   Avatar (2009). Genres: Action, Adventure, Fantasy, Science Fiction.
   In the 22nd century, a paraplegic Marine...
   Tags: culture clash, future, space war
   ```
7. Создаёт pipe-delimited поля для фильтрации в ChromaDB:
   - `genres_pipe`: `|Action|Adventure|Fantasy|`
   - `tags_pipe`: `|culture clash|future|space war|`
8. Сохраняет результат в `data/processed/movies.jsonl` (одна строка — один JSON-объект).

**Выходной формат записи:**

```json
{
  "id": "19995",
  "title": "Avatar",
  "year": 2009,
  "duration_min": 162,
  "rating": 7.2,
  "genres": "Action, Adventure, Fantasy, Science Fiction",
  "genres_pipe": "|Action|Adventure|Fantasy|Science Fiction|",
  "keywords": "culture clash, future, space war, ...",
  "tags_pipe": "|culture clash|future|space war|...|",
  "overview": "In the 22nd century, a paraplegic Marine...",
  "text_for_embedding": "Avatar (2009). Genres: Action, Adventure, ..."
}
```

---

### 7.2 Build Index (`scripts/build_index.py`)

**Назначение:** Построение векторного индекса ChromaDB из обработанных фильмов.

**Запуск:** `python scripts/build_index.py`

**Что делает:**

1. Загружает фильмы из `data/processed/movies.jsonl`.
2. Инициализирует SentenceTransformer (`all-MiniLM-L6-v2`).
3. Батчево кодирует все `text_for_embedding` (batch_size=64).
4. Создаёт коллекцию `movies` в ChromaDB с метрикой cosine.
5. Батчево добавляет документы (batch_size=500) с:
   - **ID:** id фильма
   - **Document:** text_for_embedding
   - **Embedding:** предвычисленный вектор
   - **Metadata:** title, year, duration_min, rating, genres_pipe, tags_pipe, overview
6. Выполняет тестовый запрос `"space exploration emotional drama"` для верификации.

**Важно:** Индекс нужно пересоздавать при:
- Смене embedding-модели
- Обновлении данных (re-run ingest.py)
- Изменении формата `text_for_embedding`

---

### 7.3 Evaluate (`scripts/evaluate.py`)

**Назначение:** Автоматическая оценка качества RAG-пайплайна по набору тестовых запросов.

**Запуск:** `python scripts/evaluate.py`

**Метрики:**

| Метрика | Описание | Целевое значение |
|---------|----------|-----------------|
| Recall@5 (genre) | Доля запросов, где хотя бы один фильм соответствует ожидаемому жанру | ≥ 0.75 |
| Avg Latency | Среднее время обработки запроса (мс) | ≤ 10 000 мс |
| LLM Judge | Средняя оценка рекомендаций от LLM-судьи (1-5) | ≥ 4.0 |
| Hallucination Rate | Доля запросов, где ожидались рекомендации, но система вернула "не найдено" | < 5% |

**LLM Judge** — отдельный LLM-вызов, оценивающий качество по критериям:
- Релевантность к запросу (жанр, настроение, тема)
- Разнообразие рекомендаций
- Качество объяснений

**Отказоустойчивость:**
- Каждый `rag.query()` обёрнут в `try/except` — один упавший запрос не прерывает весь evaluation.
- `evaluate_with_llm_judge()` использует `llm_call_with_retry()` с fallback-моделями.
- Между запросами — `time.sleep(1)` для снижения нагрузки на API (courtesy delay).

**Пример выходных данных:**

```
[1/30] Хочу что-то как Интерстеллар
  Genre recall: 1.00
  LLM judge: 4.5/5
  Latency: 3200ms | Type: recommendation

...

============================================================
EVALUATION RESULTS
============================================================
Recall@5 (genre):    0.85  (target: >= 0.75)
Avg Latency:         4200ms  (target: <= 10000ms)
LLM Judge:           4.2/5  (target: >= 4.0)
Hallucination rate:  3.3%  (target: < 5%)

Overall: PASS ✓
```

---

## 8. Данные

### Источник: TMDB 5000

Датасет [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) с Kaggle. Содержит ~5000 фильмов с метаданными: название, жанры, ключевые слова, синопсис, рейтинг, длительность, дата выпуска.

### Обработанный формат (`movies.jsonl`)

Каждая строка — JSON-объект с полями:

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | string | ID фильма в TMDB |
| `title` | string | Название фильма |
| `year` | int | Год выпуска |
| `duration_min` | int | Длительность в минутах |
| `rating` | float | Средняя оценка (0-10) |
| `genres` | string | Жанры через запятую |
| `genres_pipe` | string | Жанры в формате `\|Genre1\|Genre2\|` для ChromaDB |
| `keywords` | string | Ключевые слова (до 15) через запятую |
| `tags_pipe` | string | Ключевые слова в pipe-формате |
| `overview` | string | Синопсис фильма (на английском) |
| `text_for_embedding` | string | Объединённый текст для векторизации |

### Тестовый набор (`test_queries.jsonl`)

30 запросов на русском языке с разметкой:

```jsonl
{"query": "Хочу что-то как Интерстеллар", "expected_genres": ["Science Fiction", "Drama"], "expected_type": "recommendation"}
{"query": "Страшный фильм до 100 минут", "expected_genres": ["Horror"], "expected_type": "recommendation"}
{"query": "Какая сегодня погода?", "expected_genres": [], "expected_type": "off_topic"}
```

- 27 запросов типа `recommendation` (различные жанры, настроения, ограничения)
- 3 запроса типа `off_topic` (не связаны с фильмами)

---

## 9. Промпты

### Query Analyzer: system prompt

Задача: парсить запросы пользователя в структурированный JSON.

Ключевые инструкции:
- Входные данные: запрос (русский/английский) + история диалога
- **`semantic_query` всегда на английском** (эмбеддинги обучены на английском)
- Фильтры (`genre`, `mood`, `max_duration`, `min_year`, `min_rating`) только если явно указаны
- Off-topic запросы -> `{"off_topic": true}`
- Ответ строго в формате JSON, без пояснений

### Generation: system prompt

Задача: сгенерировать рекомендации на основе найденных фильмов.

Ключевые правила:
1. Рекомендовать **только** из предоставленного списка (никогда не выдумывать)
2. Отвечать **на языке пользователя**
3. Краткое объяснение для каждого фильма
4. Честно сказать, если результаты не идеально подходят
5. Лаконичные ответы

Формат ответа:

```json
{
  "movies": [
    {
      "title": "...",
      "year": 2020,
      "rating": 7.5,
      "duration_min": 120,
      "reason": "Краткое объяснение, почему фильм подходит"
    }
  ],
  "message": "Разговорное сообщение на языке пользователя"
}
```

---

## 10. Обработка ошибок и отказоустойчивость

Система спроектирована так, чтобы деградировать gracefully, а не падать.

### Уровни защиты

```
┌────────────────────────────────────────────────────────┐
│ Уровень 1: LLM retry + backoff                        │
│ RateLimitError -> повтор через 2с, 4с, 8с              │
├────────────────────────────────────────────────────────┤
│ Уровень 2: Fallback-модели                            │
│ primary -> deepseek -> gemma -> llama                    │
├────────────────────────────────────────────────────────┤
│ Уровень 3: Локальные fallback-ответы                  │
│ Query Analyzer: raw query как semantic_query           │
│ Generation: базовая инфо о фильмах без LLM            │
├────────────────────────────────────────────────────────┤
│ Уровень 4: UI error handling                          │
│ try/except -> st.error() вместо traceback              │
└────────────────────────────────────────────────────────┘
```

### Сценарии ошибок

| Ситуация | Поведение |
|----------|-----------|
| API rate limit (429) | Retry с exponential backoff -> fallback models -> локальный fallback |
| Сеть недоступна | Те же retry/fallback, после исчерпания -> сообщение об ошибке |
| LLM вернула невалидный JSON | До 2 retry JSON-парсинга -> fallback на raw data |
| LLM вернула пустой ответ | Переключение на следующую модель |
| Нет подходящих фильмов | Hallucination guard -> сообщение "попробуйте переформулировать" |
| Фильтр слишком строгий | Автоматический повтор поиска без фильтра |
| ChromaDB не создана | Ошибка при инициализации (требуется `build_index.py`) |
| Нет API-ключа | `st.error()` с инструкцией при запуске |

---

## 11. Логирование и обратная связь

### SQLite-лог (`data/logs.db`)

Каждый запрос записывается в таблицу `logs`:

| Столбец | Тип | Описание |
|---------|-----|----------|
| `request_id` | TEXT (PK) | UUID запроса |
| `timestamp` | REAL | Unix timestamp |
| `user_query` | TEXT | Исходный запрос пользователя |
| `parsed_query` | TEXT | JSON: структурированные параметры |
| `retrieved_movie_ids` | TEXT | JSON-массив ID найденных фильмов |
| `llm_response` | TEXT | Полный JSON-ответ генерации |
| `latency_ms` | REAL | Время обработки (мс) |
| `feedback` | TEXT | "like" / "dislike" / NULL |

### Обратная связь

Каждый ответ ассистента сопровождается кнопками 👍/👎. При нажатии `feedback` обновляется в соответствующей строке `logs`.

Данные логирования можно использовать для:
- Анализа популярных запросов
- Выявления проблемных паттернов (частые no_results, low judge scores)
- Корреляции feedback с качеством выдачи
- Мониторинга latency

---

## 12. Оценка качества

### Запуск

```bash
python scripts/evaluate.py
```

### Метрики и пороги

| Метрика | Формула | Целевое значение | Что измеряет |
|---------|---------|-----------------|--------------|
| **Recall@5** | (запросы с хотя бы 1 совпавшим жанром) / (всего запросов) | ≥ 0.75 | Точность жанрового поиска |
| **Avg Latency** | среднее(latency по всем запросам) | ≤ 10 с | Скорость отклика |
| **LLM Judge** | среднее(оценка LLM 1-5) | ≥ 4.0 | Общее качество рекомендаций |
| **Hallucination Rate** | (ожидали рекомендации, получили "не найдено") / total × 100% | < 5% | False negative rate |

### Общий вердикт

**PASS** — все 4 метрики достигают целевых значений одновременно.

---

## 13. Примеры работы пайплайна

### Пример 1: Жанровый запрос с ограничениями

```
Запрос: "Страшный фильм до 100 минут"

-> Query Analyzer:
  {genre: "Horror", max_duration: 100, semantic_query: "scary horror movie"}

-> Retriever:
  Фильтр: genres $contains "Horror" AND duration_min <= 100
  20 кандидатов -> реранкинг -> top-5

-> Hallucination Guard: max_similarity = 0.72 ≥ 0.4 ✓

-> Generation:
  {
    movies: [{title: "Insidious", year: 2010, rating: 6.8, duration_min: 103, reason: "..."}],
    message: "Вот несколько ужастиков, которые уложатся в ваше время:"
  }
```

### Пример 2: Off-topic запрос

```
Запрос: "Какая сегодня погода?"

-> Query Analyzer: {off_topic: true}

-> Немедленный возврат:
  {type: "off_topic", message: "Я — CineMatch, рекомендательная система фильмов..."}
```

### Пример 3: Нет подходящих результатов

```
Запрос: "Документальный фильм про выращивание сыра в Швейцарии"

-> Query Analyzer:
  {genre: "Documentary", semantic_query: "cheese making Switzerland documentary"}

-> Retriever: 3 кандидата, max_similarity = 0.25

-> Hallucination Guard: 0.25 < 0.4 ✗

-> Возврат:
  {type: "no_results", message: "По вашему запросу подходящих фильмов не найдено..."}
```

### Пример 4: Fallback при отказе API

```
Запрос: "Романтическая комедия"

-> Query Analyzer -> llm_call_with_retry:
  nvidia/nemotron -> 429 RateLimitError
    retry 1 (2с) -> 429
    retry 2 (4с) -> 429
  deepseek/deepseek-chat -> 200 OK ✓

-> Далее обычный пайплайн с ответом от deepseek
```

---
