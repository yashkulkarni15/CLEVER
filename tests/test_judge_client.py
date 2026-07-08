import pytest
from src.judge.config import JudgeConfig
from src.judge.client import DailyLimitError, JudgeClient, _retry_after_seconds


def test_from_env_reads_values():
    cfg = JudgeConfig.from_env({
        "JUDGE_API_KEY": "k", "JUDGE_BASE_URL": "http://x/v1",
        "JUDGE_MODEL": "m", "JUDGE_MAX_CONCURRENCY": "4",
        "JUDGE_TEMPERATURE": "0.0", "JUDGE_MAX_RETRIES": "3",
    })
    assert (cfg.api_key, cfg.base_url, cfg.model) == ("k", "http://x/v1", "m")
    assert cfg.max_concurrency == 4 and cfg.max_retries == 3


def test_from_env_parses_rate_limit_knobs():
    cfg = JudgeConfig.from_env({
        "JUDGE_API_KEY": "k", "JUDGE_RPM": "30",
        "JUDGE_MAX_TOKENS": "32", "JUDGE_DAILY_BACKOFF_S": "900",
    })
    assert cfg.rpm == 30.0
    assert cfg.max_tokens == 32
    assert cfg.daily_backoff_threshold_s == 900.0


def test_from_env_rate_limit_knob_defaults():
    cfg = JudgeConfig.from_env({"JUDGE_API_KEY": "k"})
    assert cfg.rpm == 25.0
    assert cfg.max_tokens == 16
    assert cfg.daily_backoff_threshold_s == 600.0


def test_from_env_requires_key():
    with pytest.raises(ValueError):
        JudgeConfig.from_env({"JUDGE_API_KEY": ""})


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def test_rate_limiter_paces_request_starts():
    clock = FakeClock()
    cfg = JudgeConfig(api_key="k", rpm=60.0)
    client = JudgeClient(
        cfg, chat_fn=lambda _m: "YES", sleep_fn=clock.sleep, time_fn=clock.time
    )
    for _ in range(3):
        assert client.complete([{"role": "user", "content": "hi"}]) == "YES"
    # first call starts immediately; 2nd and 3rd each wait one 60/rpm interval
    assert len(clock.sleeps) == 2
    assert all(abs(s - 1.0) < 1e-9 for s in clock.sleeps)


def test_rate_limiter_disabled_at_zero_rpm():
    clock = FakeClock()
    cfg = JudgeConfig(api_key="k", rpm=0.0)
    client = JudgeClient(
        cfg, chat_fn=lambda _m: "YES", sleep_fn=clock.sleep, time_fn=clock.time
    )
    for _ in range(3):
        assert client.complete([{"role": "user", "content": "hi"}]) == "YES"
    assert clock.sleeps == []


def test_no_sleep_after_final_attempt():
    cfg = JudgeConfig.from_env({
        "JUDGE_API_KEY": "k", "JUDGE_MAX_RETRIES": "3", "JUDGE_RPM": "0",
    })
    sleeps: list[float] = []

    def always_fails(messages):
        raise RuntimeError("boom")

    client = JudgeClient(cfg, chat_fn=always_fails, sleep_fn=sleeps.append)
    with pytest.raises(RuntimeError, match="boom"):
        client.complete([{"role": "user", "content": "hi"}])
    assert len(sleeps) == 2


def test_retry_after_seconds_extraction():
    class Resp:
        headers = {"retry-after": "5"}

    class HeaderExc(Exception):
        response = Resp()

    class AttrExc(Exception):
        retry_after_s = 7.5

    assert _retry_after_seconds(HeaderExc()) == 5.0
    assert _retry_after_seconds(AttrExc()) == 7.5
    assert _retry_after_seconds(RuntimeError("boom")) is None


def test_retry_after_extends_backoff_sleep():
    cfg = JudgeConfig.from_env({"JUDGE_API_KEY": "k", "JUDGE_RPM": "0"})
    sleeps: list[float] = []
    calls = {"n": 0}

    class Limited(Exception):
        retry_after_s = 5.0

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Limited("slow down")
        return "YES"

    client = JudgeClient(cfg, chat_fn=flaky, sleep_fn=sleeps.append)
    assert client.complete([{"role": "user", "content": "hi"}]) == "YES"
    assert len(sleeps) == 1
    assert sleeps[0] >= 5.0


def test_daily_cap_raises_without_sleeping():
    cfg = JudgeConfig.from_env({"JUDGE_API_KEY": "k", "JUDGE_RPM": "0"})
    sleeps: list[float] = []

    class Exhausted(Exception):
        retry_after_s = 7200.0

    def always_limited(messages):
        raise Exhausted("daily cap")

    client = JudgeClient(cfg, chat_fn=always_limited, sleep_fn=sleeps.append)
    with pytest.raises(DailyLimitError):
        client.complete([{"role": "user", "content": "hi"}])
    assert sleeps == []


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
