import importlib.util
import json
import logging
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sample_hits",
    Path(__file__).resolve().parent.parent / "scripts" / "16_sample_hits_for_judge.py",
)
sample_hits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample_hits)


def _recs(n):
    return [{"dataset": "qqp", "policy": "lfu", "distance_l2sq": i / n,
             "orig_query": f"o{i}", "new_query": f"n{i}"} for i in range(n)]


def test_sample_is_capped_and_deterministic():
    recs = _recs(500)
    a = sample_hits.stratified_sample(recs, n=100, n_bins=5, seed=7)
    b = sample_hits.stratified_sample(recs, n=100, n_bins=5, seed=7)
    assert len(a) == 100
    assert [r["new_query"] for r in a] == [r["new_query"] for r in b]  # deterministic


def test_sample_returns_all_when_fewer_than_n():
    recs = _recs(20)
    out = sample_hits.stratified_sample(recs, n=100, n_bins=5, seed=7)
    assert len(out) == 20


def test_sample_spreads_across_distance_bins():
    recs = _recs(500)
    out = sample_hits.stratified_sample(recs, n=50, n_bins=5, seed=7)
    lo = sum(1 for r in out if r["distance_l2sq"] < 0.2)
    hi = sum(1 for r in out if r["distance_l2sq"] >= 0.8)
    assert lo > 0 and hi > 0  # both tails represented


def test_topup_branch_is_deterministic_and_exact():
    recs = _recs(500)
    a = sample_hits.stratified_sample(recs, n=97, n_bins=5, seed=3)
    b = sample_hits.stratified_sample(recs, n=97, n_bins=5, seed=3)
    assert len(a) == 97
    assert [r["new_query"] for r in a] == [r["new_query"] for r in b]


def test_tied_distances_do_not_collapse_strata():
    # 60% exact-duplicate hits at distance 0.0, rest spread in (0, 1].
    recs = [{"dataset": "qqp", "policy": "lfu", "distance_l2sq": 0.0,
             "orig_query": f"o{i}", "new_query": f"n{i}"} for i in range(300)]
    recs += [{"dataset": "qqp", "policy": "lfu", "distance_l2sq": (i + 1) / 200,
              "orig_query": f"o{300 + i}", "new_query": f"n{300 + i}"} for i in range(200)]
    a = sample_hits.stratified_sample(recs, n=50, n_bins=5, seed=7)
    b = sample_hits.stratified_sample(recs, n=50, n_bins=5, seed=7)
    assert len(a) == 50
    assert [r["new_query"] for r in a] == [r["new_query"] for r in b]  # deterministic
    dists = [r["distance_l2sq"] for r in a]
    assert any(d == 0.0 for d in dists)  # tied stratum represented
    assert any(d >= 0.8 for d in dists)  # upper tail represented
    # Deduped quantile edges are [0, 0.002, 0.501, inf] → 3 effective strata
    # (zeros / lower half / upper half), quota 50 // 3 = 16 each. No stratum
    # with data may fall below its quota, and the tied mass must stay confined
    # to its own stratum (quota + at most the 2-record top-up).
    zeros = sum(1 for d in dists if d == 0.0)
    mid = sum(1 for d in dists if 0.0 < d <= 0.501)
    hi = sum(1 for d in dists if d > 0.501)
    assert zeros >= 16 and mid >= 16 and hi >= 16
    assert zeros <= 18


def test_n_below_bin_count_never_overshoots_and_rng_picks_bins():
    recs = _recs(500)
    a = sample_hits.stratified_sample(recs, n=3, n_bins=5, seed=11)
    b = sample_hits.stratified_sample(recs, n=3, n_bins=5, seed=11)
    assert len(a) == 3
    assert [r["new_query"] for r in a] == [r["new_query"] for r in b]  # deterministic
    idx = [int(r["new_query"][1:]) for r in a]
    assert idx == sorted(idx)  # returned in original-index order
    # The rng must choose WHICH bins contribute; index-sorted truncation would
    # never surface the top quintile for any seed.
    assert any(
        any(r["distance_l2sq"] >= 0.8
            for r in sample_hits.stratified_sample(recs, n=3, n_bins=5, seed=s))
        for s in range(31)
    )


def _hit(i, **overrides):
    r = {"dataset": "qqp", "policy": "lfu", "cache_size_pct": 0.10, "seed": 0,
         "workload": "default", "stream_idx": i, "distance_l2sq": i / 10,
         "matched_cache_id": i, "orig_query": f"o{i}", "new_query": f"n{i}",
         "embedding_model": "minilm"}
    r.update(overrides)
    return r


def _run_main(tmp_path, monkeypatch, lines, n=100):
    hits_dir = tmp_path / "hits"
    hits_dir.mkdir()
    (hits_dir / "hits_test.jsonl").write_text("\n".join(lines) + "\n")
    out = tmp_path / "sample.jsonl"
    monkeypatch.setattr("sys.argv", [
        "16_sample_hits_for_judge.py",
        "--hits-glob", str(hits_dir / "hits_*.jsonl"),
        "--out", str(out), "--n", str(n),
    ])
    sample_hits.main()
    return out


def test_malformed_final_line_is_skipped_and_logged(tmp_path, monkeypatch, caplog):
    lines = [json.dumps(_hit(i)) for i in range(10)]
    lines.append('{"dataset": "qqp", "poli')  # truncated final line
    with caplog.at_level(logging.WARNING):
        out = _run_main(tmp_path, monkeypatch, lines, n=5)
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(written) == 5  # valid records still sampled
    assert any("1 malformed" in m for m in caplog.messages)  # skip logged


def test_all_malformed_raises_system_exit(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, ['{"broken', "not json at all"])


def _run_main_glob(tmp_path, monkeypatch, hits_glob, out_name, n=100):
    out = tmp_path / out_name
    monkeypatch.setattr("sys.argv", [
        "16_sample_hits_for_judge.py",
        "--hits-glob", hits_glob,
        "--out", str(out), "--n", str(n),
    ])
    sample_hits.main()
    return out


def test_duplicate_hit_logs_are_deduped(tmp_path, monkeypatch, caplog):
    # Same run's hits exposed twice (e.g. re-run into a new output dir plus a
    # copied results tree) must not inflate the group or double-weight pairs.
    content = "\n".join(json.dumps(_hit(i)) for i in range(10)) + "\n"
    for sub in ("run_a", "run_b", "baseline"):
        d = tmp_path / sub / "hits"
        d.mkdir(parents=True)
        (d / "hits_test.jsonl").write_text(content)
    single_out = _run_main_glob(
        tmp_path, monkeypatch,
        str(tmp_path / "baseline" / "hits" / "hits_*.jsonl"), "single.jsonl", n=5)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        dup_out = _run_main_glob(
            tmp_path, monkeypatch,
            str(tmp_path / "run_*" / "hits" / "hits_*.jsonl"), "dup.jsonl", n=5)
    assert any("sampled from 10 hits" in m for m in caplog.messages)  # group not inflated
    assert any("10 duplicate" in m for m in caplog.messages)  # drop logged once
    assert dup_out.read_text() == single_out.read_text()  # same rng draws as single file


def test_distinct_records_across_files_are_kept_without_warning(tmp_path, monkeypatch, caplog):
    for sub, seed in (("run_a", 0), ("run_b", 1)):
        d = tmp_path / sub / "hits"
        d.mkdir(parents=True)
        (d / "hits_test.jsonl").write_text(
            "\n".join(json.dumps(_hit(i, seed=seed)) for i in range(4)) + "\n")
    with caplog.at_level(logging.INFO):
        out = _run_main_glob(
            tmp_path, monkeypatch,
            str(tmp_path / "run_*" / "hits" / "hits_*.jsonl"), "sample.jsonl")
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(written) == 8  # nothing dropped
    assert not any("duplicate" in m for m in caplog.messages)


def test_groups_key_on_full_config(tmp_path, monkeypatch, caplog):
    recs = [_hit(i) for i in range(4)]
    recs += [_hit(i + 4, cache_size_pct=0.20) for i in range(4)]
    with caplog.at_level(logging.INFO):
        out = _run_main(tmp_path, monkeypatch, [json.dumps(r) for r in recs])
    written = [json.loads(l) for l in out.read_text().splitlines()]
    # Records pass through unchanged.
    assert sorted(json.dumps(r, sort_keys=True) for r in written) == \
        sorted(json.dumps(r, sort_keys=True) for r in recs)
    group_logs = [m for m in caplog.messages if "sampled from" in m]
    assert len(group_logs) == 2  # cache_size_pct 0.10 vs 0.20 → two groups
    assert any("qqp/lfu/minilm/0.10/0/default" in m for m in group_logs)
    assert any("qqp/lfu/minilm/0.20/0/default" in m for m in group_logs)
