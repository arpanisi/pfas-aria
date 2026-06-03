"""
Provider-agnostic chat completions for agents and reporting.

Supports OpenRouter (default), Ollama (local), and RunPod (OpenAI-compatible).
"""

from __future__ import annotations

import os
import threading
from typing import Any

import requests

from src.utils.config import get_settings
from src.utils.exceptions import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)

_rate_limited_models: set[str] = set()
_rate_limited_models_lock = threading.Lock()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None and response.status_code == 429:
            return True
        body = ""
        if response is not None:
            body = response.text or ""
        return "rate limit" in body.lower() or "quota" in body.lower()
    return "rate limit" in str(exc).lower() or "quota" in str(exc).lower()


def _mark_model_rate_limited(model_id: str) -> None:
    with _rate_limited_models_lock:
        _rate_limited_models.add(model_id)


def _available_models(models: list[str]) -> list[str]:
    with _rate_limited_models_lock:
        skipped = set(_rate_limited_models)
    available = [model for model in _ordered_unique(models) if model not in skipped]
    for model in _ordered_unique(models):
        if model in skipped:
            logger.info("Skipping rate-limited OpenRouter model {}", model)
    return available


def _reset_rate_limited_models_for_tests() -> None:
    with _rate_limited_models_lock:
        _rate_limited_models.clear()


def chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    use_chat_model: bool = False,
) -> str:
    """Run a chat completion and return the assistant message text.

    use_chat_model=True selects llm.chat_model (short-form tasks like one-line rewrites).
    use_chat_model=False (default) selects llm.model (reasoning-heavy tasks).
    """
    llm = get_settings().llm
    max_t = max_tokens if max_tokens is not None else llm.max_tokens
    temp = temperature if temperature is not None else llm.temperature
    to = timeout if timeout is not None else llm.request_timeout

    provider = (llm.provider or "openrouter").lower().strip()
    if provider == "openrouter":
        return _openrouter_chat(
            messages,
            max_tokens=max_t,
            temperature=temp,
            timeout=to,
            use_chat_model=use_chat_model,
        )
    if provider == "ollama":
        return _ollama_chat(messages, max_tokens=max_t, temperature=temp, timeout=to)
    if provider == "runpod":
        return _runpod_chat(messages, max_tokens=max_t, temperature=temp, timeout=to)
    raise LLMError(f"Unknown LLM provider: {llm.provider}")


def _openrouter_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    use_chat_model: bool = False,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set (required when llm.provider is openrouter)"
        )
    llm = get_settings().llm
    base = llm.openrouter_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL", llm.openrouter_site_url or "").strip()
    site_name = os.getenv(
        "OPENROUTER_SITE_NAME", llm.openrouter_site_name or ""
    ).strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    primary = (llm.chat_model or llm.model) if use_chat_model else llm.model
    models_to_try = _available_models([primary, *llm.fallback_models])
    last_error: Exception = LLMError("No models configured")

    # Cap per-model HTTP timeout so slow/hung models don't block the fallback chain.
    # With 16+ fallbacks, each model should either respond or fail within 20s.
    # The caller's `timeout` budget applies to the whole call, not each individual attempt.
    per_model_timeout = min(timeout, 20)

    for model_id in models_to_try:
        try:
            payload: dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = requests.post(
                url, json=payload, headers=headers, timeout=per_model_timeout
            )
            response.raise_for_status()
            data = response.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content:
                # Some reasoning models return null content; the thinking text lives in
                # msg["reasoning"] (plain string) or msg["reasoning_details"][*]["text"].
                content = msg.get("reasoning") or ""
            if not content:
                for block in msg.get("reasoning_details") or []:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("thinking") or ""
                        if text:
                            content = str(text)
                            break
            if not content:
                raise LLMError(
                    f"Unexpected OpenRouter response shape: {repr(data)[:500]}"
                )
            if model_id != llm.model:
                logger.info("OpenRouter fallback succeeded with model {}", model_id)
            return content.strip()
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            if _is_rate_limit_error(e):
                logger.warning(
                    "OpenRouter model {} hit a rate limit; skipping it for "
                    "subsequent calls in this process",
                    model_id,
                )
                _mark_model_rate_limited(model_id)
            else:
                logger.warning("OpenRouter model {} failed: {}", model_id, e)
            last_error = e
            continue

    raise LLMError(
        f"All OpenRouter models failed. Last error: {last_error}"
    ) from last_error


def _ollama_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    llm = get_settings().llm
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": llm.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["message"]["content"]).strip()


def _runpod_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    import os as _os

    llm = get_settings().llm
    endpoint = (llm.runpod_endpoint or _os.getenv("RUNPOD_ENDPOINT") or "").strip()
    if not endpoint:
        raise LLMError("RUNPOD_ENDPOINT not set (required when llm.provider is runpod)")
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/chat/completions"
    response = requests.post(
        url,
        json={
            "model": llm.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"]).strip()
