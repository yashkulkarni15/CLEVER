"""Thin OpenAI-compatible chat client with retry. Works against any provider
that exposes the /v1/chat/completions schema (OpenAI, OpenRouter, Together,
Groq, local vLLM, ...)."""
import time
from typing import Callable, Optional

from src.judge.config import JudgeConfig


class JudgeClient:
    def __init__(
        self,
        config: JudgeConfig,
        chat_fn: Optional[Callable[[list[dict]], str]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self._sleep = sleep_fn
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
                    max_tokens=8,
                )
                return resp.choices[0].message.content or ""

            self._chat_fn = _default

    def complete(self, messages: list[dict]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                return self._chat_fn(messages)
            except Exception as exc:  # noqa: BLE001 — provider-agnostic retry
                last_exc = exc
                self._sleep(min(2.0 ** attempt, 30.0))
        raise last_exc  # type: ignore[misc]
