# AI HR Assistant — Resume ↔ Job Matching Agent

LLM-based система для семантического сопоставления резюме и вакансий  
с выявлением skill gaps и генерацией рекомендаций.

---

## 📌 Описание проекта

AI HR Assistant — это RAG-based система, которая:

- анализирует резюме пользователя
- семантически сопоставляет его с базой вакансий
- выделяет совпадающие и недостающие навыки
- генерирует рекомендации по улучшению резюме
- объясняет логику матчинга

---

## Команда проекта

- ML / NLP инженер — разработка retrieval + embeddings
- Backend инженер — API, интеграции
- Product / UX — пользовательский сценарий и интерфейс

---

##  Архитектура (MVP)

Pipeline:

1. Пользователь загружает резюме
2. Очистка и нормализация текста
3. Извлечение навыков
4. Embedding резюме
5. Vector search (top-K вакансий)
6. Cross-encoder rerank
7. LLM-агент:
   - сравнение требований и опыта
   - выявление skill gaps
   - генерация рекомендаций
8. Ответ пользователю (JSON + текст)

---

## Технологический стек

- Python
- FastAPI
- Streamlit
- Qdrant (Vector DB)
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
