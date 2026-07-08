"""Provider-agnostic LLM-judge configuration, sourced entirely from env/.env."""
import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class JudgeConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    max_concurrency: int = 8
    temperature: float = 0.0
    max_retries: int = 5
    rpm: float = 25.0
    max_tokens: int = 16
    daily_backoff_threshold_s: float = 600.0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "JudgeConfig":
        env = os.environ if env is None else env
        key = (env.get("JUDGE_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "JUDGE_API_KEY is not set. Copy your key into .env "
                "(see .env.example)."
            )
        return cls(
            api_key=key,
            base_url=(env.get("JUDGE_BASE_URL") or cls.base_url).strip(),
            model=(env.get("JUDGE_MODEL") or cls.model).strip(),
            max_concurrency=int(env.get("JUDGE_MAX_CONCURRENCY", cls.max_concurrency)),
            temperature=float(env.get("JUDGE_TEMPERATURE", cls.temperature)),
            max_retries=int(env.get("JUDGE_MAX_RETRIES", cls.max_retries)),
            rpm=float(env.get("JUDGE_RPM", cls.rpm)),
            max_tokens=int(env.get("JUDGE_MAX_TOKENS", cls.max_tokens)),
            daily_backoff_threshold_s=float(
                env.get("JUDGE_DAILY_BACKOFF_S", cls.daily_backoff_threshold_s)
            ),
        )
