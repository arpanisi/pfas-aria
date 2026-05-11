"""
Resilience utilities.
Circuit breaker + retry with exponential backoff for:
  - LLM calls (OpenRouter / Ollama / RunPod)
  - arXiv API
  - Semantic Scholar API

If a service is down, fail fast instead of hanging.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Retry decorators ──────────────────────────────────────────────────────────


def retry_on_network_error(
    max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0
):
    """Decorator: retry with exponential backoff on network errors."""
    return retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
    )


def retry_on_api_error(max_attempts: int = 3):
    """Decorator: retry on HTTP errors with backoff."""
    import httpx

    return retry(
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, ConnectionError)
        ),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        reraise=True,
    )


# ── Circuit breaker ───────────────────────────────────────────────────────────


class CircuitBreaker:
    """
    Simple circuit breaker.
    States: CLOSED (normal) → OPEN (failing fast) → HALF_OPEN (testing)

    Opens after N consecutive failures.
    Attempts recovery after cooldown_seconds.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._state = self.CLOSED
        self._failures: deque[float] = deque(maxlen=failure_threshold)
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if (
                self._opened_at
                and time.time() - self._opened_at > self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN (testing recovery)")
        return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function through circuit breaker."""
        state = self.state

        if state == self.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN — service appears down. "
                f"Cooldown: {self.cooldown_seconds}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    async def acall(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Async version of call."""
        state = self.state

        if state == self.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN — service appears down."
            )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
            self._failures.clear()
            logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED (recovered)")

    def _on_failure(self) -> None:
        self._failures.append(time.time())
        if len(self._failures) >= self.failure_threshold:
            if self._state != self.OPEN:
                self._state = self.OPEN
                self._opened_at = time.time()
                logger.error(
                    f"Circuit {self.name}: CLOSED → OPEN "
                    f"({self.failure_threshold} consecutive failures)"
                )


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""


# ── Singleton circuit breakers ────────────────────────────────────────────────

_llm_circuit = CircuitBreaker("llm", failure_threshold=3, cooldown_seconds=30.0)
_arxiv_circuit = CircuitBreaker("arxiv", failure_threshold=5, cooldown_seconds=60.0)
_s2_circuit = CircuitBreaker(
    "semantic_scholar", failure_threshold=5, cooldown_seconds=60.0
)


def get_llm_circuit() -> CircuitBreaker:
    return _llm_circuit


def get_arxiv_circuit() -> CircuitBreaker:
    return _arxiv_circuit


def get_s2_circuit() -> CircuitBreaker:
    return _s2_circuit


# ── Pipeline timeout ──────────────────────────────────────────────────────────


async def with_timeout(coro: Any, seconds: float, name: str = "operation") -> Any:
    """Wrap a coroutine with a timeout. Raises TimeoutError with context."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except TimeoutError:
        raise TimeoutError(
            f"{name} timed out after {seconds}s — check service availability"
        )
