"""
Provider-agnostic chat completions for agents and reporting.

Supports OpenRouter (default), Ollama (local), and RunPod (OpenAI-compatible).
"""

from __future__ import annotations

import os
from typing import Any

import requests

from src.utils.config import get_settings
from src.utils.exceptions import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> str:
    """Run a chat completion and return the assistant message text."""
    llm = get_settings().llm
    max_t = max_tokens if max_tokens is not None else llm.max_tokens
    temp = temperature if temperature is not None else llm.temperature
    to = timeout if timeout is not None else llm.request_timeout

    provider = (llm.provider or "openrouter").lower().strip()
    if provider == "openrouter":
        return _openrouter_chat(
            messages, max_tokens=max_t, temperature=temp, timeout=to
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

    payload: dict[str, Any] = {
        "model": llm.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(
            f"Unexpected OpenRouter response shape: {repr(data)[:500]}"
        ) from e


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
