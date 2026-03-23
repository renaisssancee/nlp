"""Query Analyzer: parse user queries into structured search parameters via OpenRouter."""

import json
import re
from pathlib import Path

from openai import OpenAI
import yaml

from src.llm_utils import llm_call_with_retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

with open(PROJECT_ROOT / "prompts.yaml") as f:
    prompts = yaml.safe_load(f)


class QueryAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url=config["openrouter_base_url"],
            api_key=api_key,
        )
        self.model = config["llm_model"]
        self.fallback_models = config.get("fallback_models", [])
        self.max_retries = config["query_analyzer"]["max_retries"]
        self.backoff_base = config["query_analyzer"].get("backoff_base_seconds", 2)
        self.system_prompt = prompts["query_analyzer"]["system"]
        self.user_template = prompts["query_analyzer"]["user_template"]

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown code fences and extract JSON."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def analyze(self, user_query: str, history: list[dict] | None = None) -> dict:
        """Analyze user query and return structured search parameters."""
        history_str = ""
        if history:
            last_turns = history[-(config["generation"]["history_turns"] * 2):]
            history_str = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in last_turns
            )

        user_msg = self.user_template.format(
            user_query=user_query,
            history=history_str or "None",
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        fallback = {
            "genre": None, "mood": None, "max_duration": None,
            "min_year": None, "min_rating": None, "semantic_query": user_query,
        }

        for attempt in range(self.max_retries + 1):
            raw = llm_call_with_retry(
                self.client, self.model, messages,
                fallback_models=self.fallback_models,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
            if raw is None:
                print("Query analyzer: all models failed, fallback to raw query")
                return fallback

            try:
                cleaned = self._clean_json_response(raw)
                parsed = json.loads(cleaned)

                if "off_topic" in parsed:
                    return {"off_topic": True}

                if "semantic_query" not in parsed or not parsed["semantic_query"]:
                    parsed["semantic_query"] = user_query

                return {
                    "genre": parsed.get("genre"),
                    "mood": parsed.get("mood"),
                    "max_duration": parsed.get("max_duration"),
                    "min_year": parsed.get("min_year"),
                    "min_rating": parsed.get("min_rating"),
                    "semantic_query": parsed["semantic_query"],
                }

            except (json.JSONDecodeError, KeyError) as e:
                if attempt < self.max_retries:
                    print(f"Query analyzer JSON retry {attempt + 1}: {e}")
                    continue
                print(f"Query analyzer fallback to raw query: {e}")
                return fallback
