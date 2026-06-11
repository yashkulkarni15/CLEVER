#!/bin/bash
#SBATCH --job-name=clever-p6-embed-lmsys
# NOTE: stats_dept1/stats_dept2 have NO GPU allocation (AssocGrpGRES if used
# for GPU jobs). Use engin1 for GPU (gpu=2, cpu=36, mem=180G).
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p6_embed_lmsys_%j.log
#SBATCH --error=logs/p6_embed_lmsys_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# ──────────────────────────────────────────────────────────────
# Phase 6 prerequisite: bring LMSYS into the Option-A nested layout so the
# full-matrix job uses uniform resolver paths for all 3 datasets × 2 models.
#
# LMSYS predates the nested layout: its 100k artifacts live at flat legacy
# paths (data/100k_queries.parquet, results/embeddings/100k_embeddings.npy).
# We symlink those into the nested layout (no re-encode for MiniLM) and
# GPU-encode only the missing gte-base embeddings.
#
# RNG note: the legacy run requested --sizes 10k,50k,100k,500k; the sampler
# draws one rng.choice per size in ascending order from a single
# RandomState(42), so the 100k indices were the THIRD draw. Passing
# --sizes 10k,50k,100k reproduces those exact indices (500k came after
# 100k and cannot affect it). Verified locally before submission.
#
# PRE-FETCH MODEL ONCE on a login node (compute nodes may lack internet):
#   source venv/bin/activate
#   python -c "from sentence_transformers import SentenceTransformer as S; \
#     S('thenlper/gte-base')"
# ──────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs

echo "=== [$(date)] Step 1: symlink legacy LMSYS artifacts into nested layout ==="
mkdir -p data/lmsys results/embeddings/lmsys/all-MiniLM-L6-v2
ln -sfn ../100k_queries.parquet data/lmsys/100k_queries.parquet
ln -sfn ../../100k_embeddings.npy \
    results/embeddings/lmsys/all-MiniLM-L6-v2/100k_embeddings.npy
ls -la data/lmsys/ results/embeddings/lmsys/all-MiniLM-L6-v2/

echo "=== [$(date)] Step 2: GPU-encode lmsys × gte-base ==="
python scripts/02_generate_embeddings.py \
    --input data/processed_queries.parquet \
    --dataset lmsys --model thenlper/gte-base \
    --device cuda --batch-size 512 \
    --sizes 10k,50k,100k \
    --skip-existing

echo "=== [$(date)] Step 3: verify alignment with the legacy 100k subset ==="
python - <<'EOF'
import numpy as np
import pandas as pd
from src.embeddings.encoder import QueryEncoder

emb = np.load("results/embeddings/lmsys/gte-base/100k_embeddings.npy")
q = pd.read_parquet("data/lmsys/100k_queries.parquet")
assert emb.shape == (100000, 768), f"bad shape: {emb.shape}"
assert len(q) == 100000, f"bad parquet rows: {len(q)}"
norms = np.linalg.norm(emb, axis=1)
assert np.allclose(norms, 1.0, atol=1e-4), "embeddings not unit-norm"

# Row-level alignment proof: re-encode the first 4 parquet queries and
# compare against the saved subset rows. Catches any RNG-order divergence.
enc = QueryEncoder(model_name="thenlper/gte-base", device="cpu")
probe = enc.encode(q["query_text"].head(4).tolist(), batch_size=4, normalize=True)
cos = (probe * emb[:4]).sum(axis=1)
print("alignment cosines (expect ~1.0):", np.round(cos, 6))
assert (cos > 0.999).all(), f"MISALIGNED: {cos}"
print("OK: lmsys gte-base 100k aligned with legacy subset")
EOF

echo "=== [$(date)] Done. Nested LMSYS artifacts: ==="
find results/embeddings/lmsys data/lmsys -maxdepth 2 | sort
