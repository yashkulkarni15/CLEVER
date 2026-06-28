# tests/test_judge_runner.py
import importlib.util
from pathlib import Path
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
