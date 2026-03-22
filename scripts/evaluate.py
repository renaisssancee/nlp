"""Evaluation script: measure RAG pipeline quality metrics."""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
import yaml

from src.llm_utils import llm_call_with_retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

from src.rag import CineMatchRAG

TEST_QUERIES_FILE = PROJECT_ROOT / "data" / "test_queries.jsonl"


def load_test_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def evaluate_genre_recall(result: dict, expected_genres: list[str]) -> float:
    """Check if recommended movies match expected genres."""
    if not result.get("movies") or not expected_genres:
        return 0.0
    hits = 0
    for movie in result["movies"]:
        title = movie.get("title", "").lower()
        if any(g.lower() in str(result).lower() for g in expected_genres):
            hits += 1
            break
    return 1.0 if hits > 0 else 0.0


def evaluate_with_llm_judge(query: str, result: dict, client: OpenAI) -> float:
    """Use LLM judge to score recommendation quality 1-5."""
    movies_str = json.dumps(result.get("movies", []), ensure_ascii=False, indent=2)
    prompt = f"""Rate the quality of these movie recommendations on a scale of 1-5.

User query: {query}
Recommendations: {movies_str}

Criteria:
- Relevance to the query (genre, mood, theme)
- Diversity of recommendations
- Quality of explanations

Respond with ONLY a single number 1-5."""

    raw = llm_call_with_retry(
        client,
        config["llm_model"],
        [{"role": "user", "content": prompt}],
        fallback_models=config.get("fallback_models", []),
        max_retries=config["generation"]["max_retries"],
        backoff_base=config["generation"].get("backoff_base_seconds", 2),
    )
    if raw is None:
        print("  LLM judge: all models failed, defaulting to 3.0")
        return 3.0

    try:
        score = float(raw.strip().split()[0])
        return min(max(score, 1.0), 5.0)
    except (ValueError, IndexError) as e:
        print(f"  LLM judge parse error: {e}")
        return 3.0


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    if not TEST_QUERIES_FILE.exists():
        print(f"ERROR: {TEST_QUERIES_FILE} not found")
        sys.exit(1)

    print("Initializing RAG pipeline...")
    rag = CineMatchRAG(api_key)

    judge_client = OpenAI(
        base_url=config["openrouter_base_url"],
        api_key=api_key,
    )

    test_queries = load_test_queries(TEST_QUERIES_FILE)
    print(f"Loaded {len(test_queries)} test queries\n")

    metrics = {
        "recall_scores": [],
        "latencies": [],
        "llm_judge_scores": [],
        "hallucination_count": 0,
        "total": len(test_queries),
    }

    for i, tq in enumerate(test_queries, 1):
        query = tq["query"]
        expected_genres = tq.get("expected_genres", [])
        expected_type = tq.get("expected_type", "recommendation")

        print(f"[{i}/{len(test_queries)}] {query}")

        start = time.time()
        try:
            result = rag.query(query)
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            time.sleep(1)
            continue
        latency = (time.time() - start) * 1000
        metrics["latencies"].append(latency)

        if result["type"] != expected_type:
            if expected_type == "recommendation" and result["type"] == "no_results":
                metrics["hallucination_count"] += 1
                print(f"  MISS: expected recommendations, got no_results")

        if expected_genres and result["type"] == "recommendation":
            recall = evaluate_genre_recall(result, expected_genres)
            metrics["recall_scores"].append(recall)
            print(f"  Genre recall: {recall:.2f}")

        if result["type"] == "recommendation":
            judge_score = evaluate_with_llm_judge(query, result, judge_client)
            metrics["llm_judge_scores"].append(judge_score)
            print(f"  LLM judge: {judge_score:.1f}/5")

        print(f"  Latency: {latency:.0f}ms | Type: {result['type']}")
        print()
        time.sleep(1)  # courtesy delay between API calls

    # Summary
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    avg_recall = (
        sum(metrics["recall_scores"]) / len(metrics["recall_scores"])
        if metrics["recall_scores"] else 0
    )
    avg_latency = (
        sum(metrics["latencies"]) / len(metrics["latencies"])
        if metrics["latencies"] else 0
    )
    avg_judge = (
        sum(metrics["llm_judge_scores"]) / len(metrics["llm_judge_scores"])
        if metrics["llm_judge_scores"] else 0
    )
    hallucination_rate = metrics["hallucination_count"] / metrics["total"] * 100

    print(f"Recall@5 (genre):    {avg_recall:.2f}  (target: >= 0.75)")
    print(f"Avg Latency:         {avg_latency:.0f}ms  (target: <= 10000ms)")
    print(f"LLM Judge:           {avg_judge:.1f}/5  (target: >= 4.0)")
    print(f"Hallucination rate:  {hallucination_rate:.1f}%  (target: < 5%)")
    print()

    passed = all([
        avg_recall >= 0.75,
        avg_latency <= 10000,
        avg_judge >= 4.0,
        hallucination_rate < 5,
    ])
    print(f"Overall: {'PASS ✓' if passed else 'FAIL ✗'}")


if __name__ == "__main__":
    main()
