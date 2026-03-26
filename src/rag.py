"""RAG Pipeline orchestrator: ties together query analysis, retrieval, and generation."""

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

from openai import OpenAI
import yaml

from src.llm_utils import llm_call_with_retry

from src.hallucination import check_retrieval_quality
from src.query_analyzer import QueryAnalyzer
from src.retrieval import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

with open(PROJECT_ROOT / "prompts.yaml") as f:
    prompts = yaml.safe_load(f)


class CineMatchRAG:
    def __init__(self, api_key: str):
        self.query_analyzer = QueryAnalyzer(api_key)
        self.retriever = Retriever()

        self.client = OpenAI(
            base_url=config["openrouter_base_url"],
            api_key=api_key,
        )
        self.model = config["llm_model"]
        self.fallback_models = config.get("fallback_models", [])
        self.max_retries = config["generation"]["max_retries"]
        self.backoff_base = config["generation"].get("backoff_base_seconds", 2)
        self.history_turns = config["generation"]["history_turns"]

        self.gen_system = prompts["generation"]["system"]
        self.gen_user_template = prompts["generation"]["user_template"]

        self.db_path = PROJECT_ROOT / config["logging"]["db_path"]
        self._init_db()

    def _init_db(self):
        """Initialize SQLite logging database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                request_id TEXT PRIMARY KEY,
                timestamp REAL,
                user_query TEXT,
                parsed_query TEXT,
                retrieved_movie_ids TEXT,
                llm_response TEXT,
                latency_ms REAL,
                feedback TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _log(self, request_id: str, user_query: str, parsed_query: dict,
             movie_ids: list[str], response: str, latency_ms: float):
        """Log request to SQLite."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT INTO logs (request_id, timestamp, user_query, parsed_query, "
                "retrieved_movie_ids, llm_response, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    time.time(),
                    user_query,
                    json.dumps(parsed_query, ensure_ascii=False),
                    json.dumps(movie_ids),
                    response,
                    latency_ms,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Logging error: {e}")

    def save_feedback(self, request_id: str, feedback: str):
        """Save user feedback for a request."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "UPDATE logs SET feedback = ? WHERE request_id = ?",
                (feedback, request_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Feedback save error: {e}")

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown code fences and extract JSON."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def _generate_response(self, user_query: str, movies: list[dict],
                           history: list[dict] | None) -> dict:
        """Call LLM to generate final recommendation response."""
        history_str = ""
        if history:
            last_turns = history[-(self.history_turns * 2):]
            history_str = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in last_turns
            )

        movies_json = json.dumps(
            [
                {
                    "title": m["title"],
                    "year": m["year"],
                    "rating": m["rating"],
                    "duration_min": m["duration_min"],
                    "genres": m["genres"],
                    "overview": m["overview"],
                }
                for m in movies
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_msg = self.gen_user_template.format(
            user_query=user_query,
            movies_json=movies_json,
            history=history_str or "None",
        )

        messages = [
            {"role": "system", "content": self.gen_system},
            {"role": "user", "content": user_msg},
        ]

        fallback_result = {
            "movies": [
                {
                    "title": m["title"],
                    "year": m["year"],
                    "rating": m["rating"],
                    "duration_min": m["duration_min"],
                    "reason": m.get("overview", "")[:100],
                }
                for m in movies[:5]
            ],
            "message": "Вот что я нашёл по вашему запросу:",
        }

        for attempt in range(self.max_retries + 1):
            raw = llm_call_with_retry(
                self.client, self.model, messages,
                fallback_models=self.fallback_models,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
            if raw is None:
                print("Generation: all models failed, using fallback")
                return fallback_result

            try:
                cleaned = self._clean_json_response(raw)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                if attempt < self.max_retries:
                    print(f"Generation JSON retry {attempt + 1}: {e}")
                    continue
                print(f"Generation fallback: {e}")
                return fallback_result

    def query(self, user_query: str, history: list[dict] | None = None) -> dict:
        """Main entry point: process user query and return recommendations."""
        request_id = str(uuid.uuid4())
        start = time.time()

        parsed = self.query_analyzer.analyze(user_query, history)

        if parsed.get("off_topic"):
            latency = (time.time() - start) * 1000
            self._log(request_id, user_query, parsed, [], "off_topic", latency)
            return {
                "request_id": request_id,
                "type": "off_topic",
                "message": "Я — CineMatch, рекомендательная система фильмов. "
                           "Задайте вопрос о фильмах, и я помогу подобрать что-то интересное!",
                "movies": [],
            }

        candidates = self.retriever.retrieve(parsed)

        is_ok, fallback_msg = check_retrieval_quality(candidates)
        if not is_ok:
            latency = (time.time() - start) * 1000
            self._log(request_id, user_query, parsed, [], fallback_msg, latency)
            return {
                "request_id": request_id,
                "type": "no_results",
                "message": fallback_msg,
                "movies": [],
            }

        gen_result = self._generate_response(user_query, candidates, history)
        latency = (time.time() - start) * 1000

        movie_ids = [c["id"] for c in candidates]
        self._log(request_id, user_query, parsed, movie_ids,
                  json.dumps(gen_result, ensure_ascii=False), latency)

        return {
            "request_id": request_id,
            "type": "recommendation",
            "message": gen_result.get("message", ""),
            "movies": gen_result.get("movies", []),
        }
