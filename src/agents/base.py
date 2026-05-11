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

from src.utils.config import get_settings
from src.utils.exceptions import LLMError
from src.utils.llm_client import chat_completion
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

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            return chat_completion(
                messages,
                max_tokens=self.llm_cfg.max_tokens,
                temperature=self.llm_cfg.temperature,
                timeout=self.llm_cfg.request_timeout,
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"[{self.name}] LLM call failed: {e}") from e

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
