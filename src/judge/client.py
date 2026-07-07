"""Thin OpenAI-compatible chat client with retry. Works against any provider
that exposes the /v1/chat/completions schema (OpenAI, OpenRouter, Together,
Groq, local vLLM, ...). Reasoning-style models that require
``max_completion_tokens`` instead of ``max_tokens`` are out of scope."""
import threading
import time
from typing import Callable, Optional

from src.judge.config import JudgeConfig


class DailyLimitError(Exception):
    """Provider Retry-After exceeds the daily-backoff threshold: the daily
    quota is exhausted, not the per-minute one, so retrying now is pointless."""


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Retry-After seconds from an ``openai.RateLimitError``-shaped exception
    (``exc.response.headers``) or a duck-typed ``retry_after_s`` attribute."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        value = headers.get("retry-after")
        if value is not None:
            return float(value)
    value = getattr(exc, "retry_after_s", None)
    if value is not None:
        return float(value)
    return None


class JudgeClient:
    def __init__(
        self,
        config: JudgeConfig,
        chat_fn: Optional[Callable[[list[dict]], str]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._sleep = sleep_fn
        self._time = time_fn
        self._pace_lock = threading.Lock()
        self._next_start: Optional[float] = None
        if chat_fn is not None:
            self._chat_fn = chat_fn
        else:
            from openai import OpenAI

            client = OpenAI(api_key=config.api_key, base_url=config.base_url)

            def _default(messages: list[dict]) -> str:
                resp = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return resp.choices[0].message.content or ""

            self._chat_fn = _default

    def _await_turn(self) -> None:
        # Min-interval pacing: each request start is scheduled 60/rpm seconds
        # after the previously scheduled start, so concurrent threads queue up
        # behind the lock instead of bursting.
        if self.config.rpm <= 0:
            return
        interval = 60.0 / self.config.rpm
        with self._pace_lock:
            now = self._time()
            start = now if self._next_start is None else max(now, self._next_start)
            self._next_start = start + interval
            wait = start - now
        if wait > 0:
            self._sleep(wait)

    def complete(self, messages: list[dict]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            self._await_turn()
            try:
                return self._chat_fn(messages)
            except Exception as exc:  # noqa: BLE001 — provider-agnostic retry
                last_exc = exc
                retry_after = _retry_after_seconds(exc)
                if (
                    retry_after is not None
                    and retry_after > self.config.daily_backoff_threshold_s
                ):
                    raise DailyLimitError(
                        f"provider asked to retry after {retry_after:.0f}s, above "
                        f"the {self.config.daily_backoff_threshold_s:.0f}s daily "
                        "backoff threshold; re-run after the quota resets — the "
                        "verdict cache resumes where it left off"
                    ) from exc
                if attempt + 1 < self.config.max_retries:
                    backoff = min(2.0 ** attempt, 30.0)
                    self._sleep(min(max(retry_after or 0.0, backoff), 60.0))
        raise last_exc  # type: ignore[misc]
