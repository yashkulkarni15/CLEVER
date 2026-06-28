import pytest
from src.judge.config import JudgeConfig
from src.judge.client import JudgeClient


def test_from_env_reads_values():
    cfg = JudgeConfig.from_env({
        "JUDGE_API_KEY": "k", "JUDGE_BASE_URL": "http://x/v1",
        "JUDGE_MODEL": "m", "JUDGE_MAX_CONCURRENCY": "4",
        "JUDGE_TEMPERATURE": "0.0", "JUDGE_MAX_RETRIES": "3",
    })
    assert (cfg.api_key, cfg.base_url, cfg.model) == ("k", "http://x/v1", "m")
    assert cfg.max_concurrency == 4 and cfg.max_retries == 3


def test_from_env_requires_key():
    with pytest.raises(ValueError):
        JudgeConfig.from_env({"JUDGE_API_KEY": ""})


def test_client_retries_then_succeeds():
    cfg = JudgeConfig.from_env({"JUDGE_API_KEY": "k", "JUDGE_MAX_RETRIES": "5"})
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate limit")
        return "YES"

    client = JudgeClient(cfg, chat_fn=flaky, sleep_fn=lambda _s: None)
    assert client.complete([{"role": "user", "content": "hi"}]) == "YES"
    assert calls["n"] == 3
