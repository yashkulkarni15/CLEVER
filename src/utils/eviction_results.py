"""Shared loader for aggregated eviction-result JSONs.

The dataset for each file is read from the manifest command line stored inside
the JSON (the token after --dataset), never inferred from directory names, so
results stay correctly attributed no matter where the files live.
"""
import glob
import json
import shlex
from pathlib import Path


def load_aggregated_hit_rates(glob_pattern: str, cache_key: str) -> dict:
    """Map (dataset, policy) → aggregated hit_rate_mean for every file matching
    glob_pattern. `cache_key` must match the aggregated block exactly (e.g.
    "0.10"); any ambiguity raises SystemExit rather than guessing."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"no eviction result files matched: {glob_pattern}")
    out: dict = {}
    sources: dict = {}
    for f in files:
        data = json.loads(Path(f).read_text())
        tokens = shlex.split(data.get("manifest", {}).get("command_line", ""))
        try:
            dataset = tokens[tokens.index("--dataset") + 1]
        except (ValueError, IndexError):
            raise SystemExit(
                f"{f}: manifest.command_line has no --dataset flag; cannot "
                "attribute these results to a dataset"
            ) from None
        for policy, block in data.get("aggregated", {}).items():
            if cache_key not in block:
                raise SystemExit(
                    f"{f}: cache key {cache_key!r} not found in "
                    f"aggregated[{policy!r}] (available: {sorted(block)})"
                )
            if (dataset, policy) in out:
                raise SystemExit(
                    f"duplicate results for (dataset={dataset!r}, "
                    f"policy={policy!r}) at cache key {cache_key!r}: "
                    f"{sources[(dataset, policy)]} and {f} both match the "
                    "glob; narrow the pattern or remove the stray file"
                )
            out[(dataset, policy)] = block[cache_key]["hit_rate_mean"]
            sources[(dataset, policy)] = f
    return out
