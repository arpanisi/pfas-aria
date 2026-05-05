"""
Base Agent.
All specialist agents inherit from this. Provides shared LLM access,
structured output parsing, and consistent logging/error handling.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import requests

from src.utils.config import get_settings
from src.utils.exceptions import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all PFAS-ARIA specialist agents."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm_cfg = self.settings.llm
        self.name = self.__class__.__name__

    # ── LLM interface ────────────────────────────────────────────────────────

    def call_llm(self, prompt: str, system: str | None = None) -> str:
        """Call the configured LLM and return the response text."""
        logger.debug(f"[{self.name}] LLM call — prompt length: {len(prompt)}")

        try:
            if self.llm_cfg.provider == "ollama":
                return self._call_ollama(prompt, system)
            elif self.llm_cfg.provider == "runpod":
                return self._call_runpod(prompt, system)
            else:
                raise LLMError(f"Unknown LLM provider: {self.llm_cfg.provider}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"[{self.name}] LLM call failed: {e}") from e

    def _call_ollama(self, prompt: str, system: str | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.llm_cfg.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.llm_cfg.temperature,
                    "num_predict": self.llm_cfg.max_tokens,
                },
            },
            timeout=self.llm_cfg.request_timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _call_runpod(self, prompt: str, system: str | None) -> str:
        import os
        endpoint = self.llm_cfg.runpod_endpoint or os.getenv("RUNPOD_ENDPOINT")
        if not endpoint:
            raise LLMError("RUNPOD_ENDPOINT not set in .env")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{endpoint}/chat/completions",
            json={
                "model": self.llm_cfg.model,
                "messages": messages,
                "temperature": self.llm_cfg.temperature,
                "max_tokens": self.llm_cfg.max_tokens,
            },
            timeout=self.llm_cfg.request_timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ── JSON parsing ─────────────────────────────────────────────────────────

    def parse_json(self, text: str) -> Any:
        """Extract and parse JSON from LLM response.
        Handles markdown code blocks and raw JSON."""
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object or array in the text
        for pattern in [r"\{.*\}", r"\[.*\]"]:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        raise LLMError(
            f"[{self.name}] Could not parse JSON from LLM response:\n{text[:500]}"
        )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Each agent implements its own run() method."""
        ...
