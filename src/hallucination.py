"""Hallucination guard: check retrieval quality before calling LLM."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)


SIMILARITY_THRESHOLD = config["retrieval"]["similarity_threshold"]


def check_retrieval_quality(candidates: list[dict]) -> tuple[bool, str]:
    """Check if retrieval results are good enough to generate a response.

    Returns:
        (is_ok, message): is_ok=True if quality sufficient, message for fallback.
    """
    if not candidates:
        return False, "По вашему запросу подходящих фильмов не найдено. Попробуйте переформулировать запрос."

    best_similarity = max(c.get("similarity", 0) for c in candidates)
    if best_similarity < SIMILARITY_THRESHOLD:
        return False, (
            "По вашему запросу подходящих фильмов не найдено. "
            "Попробуйте описать желаемый фильм другими словами."
        )

    return True, ""
