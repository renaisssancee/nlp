"""Shared LLM call helper with retry and fallback model support."""

import time

import openai


def llm_call_with_retry(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    fallback_models: list[str] | None = None,
    max_retries: int = 2,
    backoff_base: float = 2.0,
) -> str | None:
    """Call LLM with exponential backoff retry and fallback models.

    Returns raw response text or None if all models/retries exhausted.
    """
    models_to_try = [model] + (fallback_models or [])

    for current_model in models_to_try:
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                )
                content = response.choices[0].message.content if response.choices else None
                if not content:
                    raise ValueError("Empty response from LLM")
                return content
            except (openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError) as e:
                print(f"LLM error ({current_model}, attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    sleep_time = backoff_base ** (attempt + 1)
                    print(f"  Retrying in {sleep_time:.0f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"  Exhausted retries for {current_model}, trying next model...")
                    break
            except ValueError as e:
                print(f"LLM error ({current_model}): {e}")
                break  # empty response — try next model

    print("All models exhausted.")
    return None
