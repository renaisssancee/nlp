# AI Movie Assistant — Film Recommendation and Discovery Agent

LLM-based система для семантического поиска фильмов по описанию настроения,  
жанра и предпочтений с объяснением рекомендаций.

---

## Описание проекта

AI Movie Assistant — это RAG-based система, которая:

- принимает запрос пользователя на естественном языке («хочу что-то грустное про войну»)
- семантически сопоставляет запрос с базой фильмов
- объясняет, почему именно эти фильмы подходят
- выделяет ключевые темы, жанры и настроение
- генерирует персональные рекомендации с обоснованием

---

## Команда проекта

- **ML / NLP инженер** — разработка retrieval + embeddings (Пахолкова Мария)
- **Backend инженер** — API, интеграции (Цисарук Мария)
- **Product / UX** — пользовательский сценарий и интерфейс (Смешкова Екатерина)

---

## Данные

| Датасет | Источник | Размер | Назначение |
|---|---|---|---|
| IMDB Top 1000 Movies | Kaggle | около 1 000 фильмов | Основная база (RAG) |
| IMDB Movie Reviews | Kaggle | около 50 000 рецензий | Обогащение контекста |
| MovieLens Dataset | grouplens.org | около 60 000 фильмов | Расширение базы |

---

## Архитектура (MVP)

Pipeline:

1. Пользователь вводит запрос в свободной форме
2. Очистка и нормализация текста запроса
3. Embedding запроса (bi-encoder)
4. Vector search — top-K фильмов из базы
5. Cross-encoder rerank
6. LLM-агент:
   - анализ соответствия запроса и описания фильма
   - объяснение логики рекомендации
   - генерация краткого обзора почему фильм подходит
7. Ответ пользователю (JSON + текст)

---

## Технологический стек

- **Python**
- **FastAPI**
- **Streamlit**
- **Qdrant** (Vector DB)
- **Sentence-Transformers** (bi-encoder)
- **Cross-Encoder** (rerank)
- **GPT-4o mini / Gemini** (LLM)

---

## Метрики

**Retrieval:**
- Recall@K
- nDCG@K
- Precision@K

**LLM:**
- groundedness (ответ основан на данных из базы)
- структурная валидность JSON
- ручная проверка релевантности рекомендаций

**UX:**
- latency
- пользовательская оценка (like/dislike)- Qdrant (Vector DB)
- Sentence-Transformers (bi-encoder)
- Cross-Encoder (rerank)
- GPT-4o mini / Gemini (LLM)

---

## Метрики

Retrieval:
- Recall@K
- nDCG@K
- Precision@K

LLM:
- groundedness
- структурная валидность JSON
- ручная проверка рекомендаций

UX:
- latency
- пользовательская оценка (like/dislike)

---
