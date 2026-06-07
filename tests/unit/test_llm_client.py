"""Tests for LLM fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import requests

from src.utils import llm_client


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="openrouter",
            model="provider/rate-limited",
            chat_model=None,
            fallback_models=["provider/fallback"],
            max_tokens=128,
            temperature=0.1,
            request_timeout=30,
            openrouter_base_url="https://openrouter.test/api/v1",
            openrouter_site_url=None,
            openrouter_site_name=None,
        )
    )


def _rate_limit_response() -> requests.Response:
    response = requests.Response()
    response.status_code = 429
    response.url = "https://openrouter.test/api/v1/chat/completions"
    response._content = b'{"error":"rate limit exceeded"}'
    return response


def _success_response() -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"choices":[{"message":{"content":"fallback ok"}}]}'
    return response


def test_rate_limited_openrouter_model_is_skipped_on_later_calls(monkeypatch):
    llm_client._reset_rate_limited_models_for_tests()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    calls: list[str] = []

    def fake_post(_url, *, json, headers, timeout):  # noqa: ANN001
        calls.append(json["model"])
        if json["model"] == "provider/rate-limited":
            return _rate_limit_response()
        return _success_response()

    with (
        patch("src.utils.llm_client.get_settings", return_value=_settings()),
        patch("src.utils.llm_client.requests.post", side_effect=fake_post),
    ):
        first = llm_client.chat_completion([{"role": "user", "content": "hello"}])
        second = llm_client.chat_completion([{"role": "user", "content": "again"}])

    assert first == "fallback ok"
    assert second == "fallback ok"
    assert calls == [
        "provider/rate-limited",
        "provider/fallback",
        "provider/fallback",
    ]

    llm_client._reset_rate_limited_models_for_tests()
