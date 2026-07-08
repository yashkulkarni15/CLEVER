# tests/test_judge_runner.py
import importlib.util
import json
import threading
import time
from concurrent.futures import wait as futures_wait
from pathlib import Path

import pytest

from src.judge.config import JudgeConfig
from src.judge.client import JudgeClient

SPEC = importlib.util.spec_from_file_location(
    "run_judge",
    Path(__file__).resolve().parent.parent / "scripts" / "17_run_llm_judge.py",
)
run_judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_judge)


def test_unique_pairs_judged_once_and_cached():
    cfg = JudgeConfig.from_env({"JUDGE_API_KEY": "k"})
    calls = {"n": 0}

    def fake(messages):
        calls["n"] += 1
        # YES when the new query mentions "everest", else NO.
        return "YES" if "everest" in messages[-1]["content"].lower() else "NO"

    client = JudgeClient(cfg, chat_fn=fake, sleep_fn=lambda _s: None)
    records = [
        {"dataset": "qqp", "policy": "lfu", "orig_query": "Everest height?",
         "new_query": "How tall is Everest?"},
        {"dataset": "qqp", "policy": "arc", "orig_query": "Everest height?",
         "new_query": "How tall is Everest?"},   # duplicate pair → cache hit
        {"dataset": "qqp", "policy": "lfu", "orig_query": "capital of France?",
         "new_query": "Best pizza in Rome?"},
    ]
    cache = {}
    verdicts = run_judge.judge_samples(records, client, cache, max_workers=2)
    assert calls["n"] == 2                      # only 2 unique pairs
    k_dup = run_judge.pair_key("Everest height?", "How tall is Everest?")
    assert verdicts[k_dup] is True

    scores = run_judge.quality_adjusted(
        records, verdicts, raw_hit_rates={("qqp", "lfu"): 0.5543, ("qqp", "arc"): 0.5543},
    )
    # lfu: 1 YES of 2 → mean 0.5 → 0.5543*0.5
    assert abs(scores[("qqp", "lfu")]["quality_adjusted_hit_rate"] - 0.5543 * 0.5) < 1e-9
    assert scores[("qqp", "arc")]["mean_yes"] == 1.0


def _cfg(**extra):
    env = {"JUDGE_API_KEY": "k", "JUDGE_RPM": "0"}
    env.update(extra)
    return JudgeConfig.from_env(env)


def _rec(policy, new_query, orig_query="What is X?", **extra):
    return {"dataset": "qqp", "policy": policy,
            "orig_query": orig_query, "new_query": new_query, **extra}


def _cache_lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def test_failed_pair_does_not_abort_and_others_persist_mid_run(tmp_path):
    cache_path = tmp_path / "verdict_cache.jsonl"
    records = [_rec("lfu", "good pair one?"),
               _rec("lfu", "good pair two?"),
               _rec("lfu", "bad-pair three?")]
    k_good1 = run_judge.pair_key("What is X?", "good pair one?")
    k_good2 = run_judge.pair_key("What is X?", "good pair two?")
    k_bad = run_judge.pair_key("What is X?", "bad-pair three?")
    snapshots = []

    def fake(messages):
        content = messages[-1]["content"]
        if "bad-pair" in content:
            # Wait (bounded) for the good verdicts to land on disk, proving
            # persistence happens mid-run, not at the end.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                txt = cache_path.read_text() if cache_path.exists() else ""
                if k_good1 in txt and k_good2 in txt:
                    break
                time.sleep(0.01)
            snapshots.append(cache_path.read_text() if cache_path.exists() else "")
            raise RuntimeError("provider 500")
        return "YES"

    client = JudgeClient(_cfg(JUDGE_MAX_RETRIES="2"), chat_fn=fake,
                         sleep_fn=lambda _s: None)
    cache, stats = {}, {}
    verdicts = run_judge.judge_samples(records, client, cache, max_workers=1,
                                       cache_path=cache_path, stats=stats)
    assert verdicts[k_good1] is True and verdicts[k_good2] is True
    assert verdicts[k_bad] is None
    assert k_bad not in cache
    assert stats["n_failed"] == 1
    assert stats["status"][k_bad] == "failed"
    # Mid-run durability: both good verdicts were on disk while bad still ran.
    assert k_good1 in snapshots[0] and k_good2 in snapshots[0]
    rows = _cache_lines(cache_path)
    assert {r["key"] for r in rows} == {k_good1, k_good2}


def test_none_verdict_not_persisted_and_retried(tmp_path):
    cache_path = tmp_path / "verdict_cache.jsonl"
    records = [_rec("lfu", "garbled pair?")]
    k = run_judge.pair_key("What is X?", "garbled pair?")
    calls = {"n": 0}

    def fake(messages):
        calls["n"] += 1
        return "MAYBE" if calls["n"] == 1 else "YES"

    client = JudgeClient(_cfg(), chat_fn=fake, sleep_fn=lambda _s: None)
    cache, stats = {}, {}
    verdicts = run_judge.judge_samples(records, client, cache, max_workers=1,
                                       cache_path=cache_path, stats=stats)
    assert calls["n"] == 1
    assert verdicts[k] is None
    assert k not in cache
    assert stats["n_unparsed"] == 1
    assert _cache_lines(cache_path) == []          # None never persisted

    stats2 = {}
    verdicts2 = run_judge.judge_samples(records, client, cache, max_workers=1,
                                        cache_path=cache_path, stats=stats2)
    assert calls["n"] == 2                         # re-attempted, not skipped
    assert verdicts2[k] is True
    assert _cache_lines(cache_path) == [{"key": k, "verdict": True}]


def test_load_cache_skips_legacy_null_rows(tmp_path):
    cache_path = tmp_path / "verdict_cache.jsonl"
    cache_path.write_text(
        json.dumps({"key": "a", "verdict": True}) + "\n"
        + json.dumps({"key": "b", "verdict": None}) + "\n"
        + json.dumps({"key": "c", "verdict": False}) + "\n")
    cache = run_judge.load_cache(cache_path)
    assert cache == {"a": True, "c": False}


def test_max_calls_budget_counts_pending(tmp_path):
    records = [_rec("lfu", "pair one?"), _rec("lfu", "pair two?"),
               _rec("lfu", "pair three?")]
    keys = [run_judge.pair_key("What is X?", r["new_query"]) for r in records]
    calls = {"n": 0}

    def fake(messages):
        calls["n"] += 1
        return "YES"

    client = JudgeClient(_cfg(), chat_fn=fake, sleep_fn=lambda _s: None)
    cache, stats = {}, {}
    verdicts = run_judge.judge_samples(records, client, cache, max_workers=1,
                                       cache_path=tmp_path / "c.jsonl",
                                       max_calls=1, stats=stats)
    assert calls["n"] == 1
    assert verdicts[keys[0]] is True
    assert verdicts[keys[1]] is None and verdicts[keys[2]] is None
    assert stats["n_pending"] == 2
    assert stats["status"][keys[1]] == "pending"
    assert stats["status"][keys[2]] == "pending"


def test_daily_limit_clean_partial_return(tmp_path):
    cache_path = tmp_path / "verdict_cache.jsonl"
    records = [_rec("lfu", "fine pair?"), _rec("lfu", "quota pair?")]
    k_ok = run_judge.pair_key("What is X?", "fine pair?")
    k_quota = run_judge.pair_key("What is X?", "quota pair?")

    class _RateLimit(Exception):
        retry_after_s = 86400.0                    # above the daily threshold

    def fake(messages):
        if "quota pair" in messages[-1]["content"]:
            raise _RateLimit("daily quota exhausted")
        return "YES"

    client = JudgeClient(_cfg(), chat_fn=fake, sleep_fn=lambda _s: None)
    cache, stats = {}, {}
    verdicts = run_judge.judge_samples(records, client, cache, max_workers=1,
                                       cache_path=cache_path, stats=stats)
    assert stats["daily_limit"] is True
    assert verdicts[k_ok] is True
    assert verdicts[k_quota] is None
    assert stats["status"][k_quota] == "pending"   # retried after quota reset
    assert {r["key"] for r in _cache_lines(cache_path)} == {k_ok}


def test_quality_adjusted_reports_counters():
    records = [_rec("lfu", "one?"), _rec("lfu", "two?"), _rec("lfu", "three?"),
               _rec("lfu", "four?"), _rec("lfu", "five?"),
               _rec("arc", "six?")]
    keys = [run_judge.pair_key("What is X?", r["new_query"]) for r in records]
    verdicts = {keys[0]: True, keys[1]: False}
    statuses = {keys[0]: "judged", keys[1]: "judged", keys[2]: "unparsed",
                keys[3]: "failed", keys[4]: "pending", keys[5]: "pending"}
    scores = run_judge.quality_adjusted(
        records, verdicts, raw_hit_rates={("qqp", "lfu"): 0.5},
        statuses=statuses)
    lfu = scores[("qqp", "lfu")]
    assert lfu["n_records"] == 5
    assert lfu["n_judged"] == 2
    assert lfu["n_unparsed"] == 1
    assert lfu["n_failed"] == 1
    assert lfu["n_pending"] == 1
    assert lfu["mean_yes"] == 0.5
    assert lfu["quality_adjusted_hit_rate"] == 0.25
    arc = scores[("qqp", "arc")]                   # nothing judged yet
    assert arc["n_records"] == 1 and arc["n_judged"] == 0
    assert arc["mean_yes"] is None
    assert arc["quality_adjusted_hit_rate"] is None


def _full_rec(policy, new_query, cache_size_pct=0.10, **overrides):
    r = _rec(policy, new_query, embedding_model="all-MiniLM-L6-v2",
             cache_size_pct=cache_size_pct, seed=42, workload="temporal")
    r.update(overrides)
    return r


def test_homogeneity_mixed_cache_sizes_raises():
    records = [_full_rec("lfu", "one?", cache_size_pct=0.10),
               _full_rec("lfu", "two?", cache_size_pct=0.20)]
    with pytest.raises(SystemExit) as ei:
        run_judge.validate_homogeneity(records, "0.10")
    msg = str(ei.value)
    assert "qqp" in msg and "lfu" in msg
    assert "0.10" in msg and "0.20" in msg


def test_homogeneity_cache_pct_must_match_raw_cache_key():
    records = [_full_rec("lfu", "one?", cache_size_pct=0.20)]
    with pytest.raises(SystemExit) as ei:
        run_judge.validate_homogeneity(records, "0.10")
    assert "0.20" in str(ei.value) and "0.10" in str(ei.value)


def test_homogeneity_missing_fields_tells_user_to_regenerate():
    records = [{"dataset": "qqp", "policy": "lfu",
                "orig_query": "a?", "new_query": "b?"}]
    with pytest.raises(SystemExit) as ei:
        run_judge.validate_homogeneity(records, "0.10")
    assert "regenerate" in str(ei.value)
