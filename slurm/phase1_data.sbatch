#!/bin/bash
#SBATCH --job-name=clever-p1-data
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/p1_data_%j.log
#SBATCH --error=logs/p1_data_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# ──────────────────────────────────────────────────────────────
# Phase 1a: Build MOSS + QQP query parquets from the raw files.
# CPU job. Reads data/raw/{moss,qqp}/ (transferred via rsync) and
# writes data/<dataset>/{raw,processed}_queries.parquet + subsets.
# MOSS json.load is memory-heavy (~851 MB raw) → 32 GB allocation.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs

# MOSS turn mode. Decided 2026-06-05: "--all-turns" — single-turn-only
# yielded only 77,019 (< 100K target); --all-turns takes the first human
# turn of EVERY conversation (mirrors LMSYS methodology) → 300K+, enough to
# sample a clean 100k subset uniform with LMSYS/QQP. Revert to
# "--single-turn-only" only if you specifically want pure single-turn.
MOSS_TURNS="--all-turns"

echo "=== [$(date)] Build QQP parquet + subsets ==="
python scripts/01_download_dataset.py --dataset qqp --raw-path data/raw/qqp/

echo "=== [$(date)] Build MOSS parquet + subsets ($MOSS_TURNS) ==="
python scripts/01_download_dataset.py --dataset moss --raw-path data/raw/moss/ $MOSS_TURNS

echo "=== [$(date)] Resulting processed-query counts ==="
python - <<'PY'
import pandas as pd
for ds in ["qqp", "moss"]:
    try:
        df = pd.read_parquet(f"data/{ds}/processed_queries.parquet")
        print(f"  {ds:5s}: {len(df):>8,} processed queries")
    except Exception as e:
        print(f"  {ds:5s}: ERROR {e}")
PY
echo "=== [$(date)] Phase 1a done ==="
